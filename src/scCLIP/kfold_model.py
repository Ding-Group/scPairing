import sys
import os
import argparse
import gc
import pickle
sys.path.append('./')
sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scvi-tools/')

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import scipy
import torch
import wandb

from main import ModelName
from trainers.UnsupervisedTrainer import UnsupervisedTrainer
import scvi

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, log_loss


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient

    Source: https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor#Predicting
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def foscttm(x: np.ndarray, y: np.ndarray, split=10000, **kwargs):
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    foscttms = []
    for i in range(0, x.shape[0], split):
        x_s = x[i: i + split]
        d = scipy.spatial.distance_matrix(x_s, y, **kwargs)
        foscttm_x = (d < np.expand_dims(np.diag(d, k=i), axis=1)).mean(axis=1)
        foscttms.append(foscttm_x)
    return np.concatenate(foscttms)


def reconstruct_mod1(scvi_model):
    def f(mod2_features, true_features, counts, library_size, cell_indices, is_training, is_imputation, batch_indices=None):
        if is_training:
            return None, 0
        if batch_indices is None:
            batch_indices = torch.zeros(mod2_features.shape[0], device=mod2_features.device)
        library_size = torch.log(library_size) if not is_imputation else torch.ones((mod2_features.shape[0], 1))
        res = scvi_model.module.generative(mod2_features, library_size, batch_indices.reshape((mod2_features.shape[0], 1)))
        if is_imputation:
            return res['px'].mu, None
        loss = -res['px'].log_prob(counts).sum(-1).mean()
        return res['px'].mu, loss
    return f


def reconstruct_mod2(pvi_model):
    def f(mod1_features, true_features, counts, library_size, cell_indices, is_training, is_imputation, batch_indices=None):
        if is_training:
            return None, 0
        if batch_indices is None:
            batch_indices = torch.zeros(mod1_features.shape[0], device=mod1_features.device)
        res = pvi_model.module.generative(mod1_features, mod1_features, batch_indices.reshape((mod1_features.shape[0], 1)))
        if is_imputation:
            return res['p'], None
        dres = pvi_model.module.d_encoder(counts, batch_indices, ())
        region_factors = torch.sigmoid(pvi_model.module.region_factors)
        loss = pvi_model.module.get_reconstruction_loss(res['p'], dres, region_factors, counts).mean()
        return res['p'], loss
    return f


def main(config, seed=0):
    wandb.init(project="kfold-model", config=config)

    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None
    batch_col = trainer_params.get('batch_col', 'batch_indices')
    use_pretrained_decoder = config.get('pretrained', False)

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    del mod1_adata.uns, mod1_adata.obsp, mod2_adata.uns, mod2_adata.obsp

    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc_full_rna/embs.pkl', 'rb') as f:
        mod1_adata.obsm['X_scVI'] = pickle.load(f)

    # mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable].copy()
    sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.shape[0] * 0.01))

    if use_pretrained_decoder:
        scvi_model = scvi.model.SCVI.load(
            '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc_full_rna/',
            # '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc/',
            adata=mod1_adata,
            prefix='scvi_'
        )
        pvi_model = scvi.model.PEAKVI.load(
            '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc',
            adata=mod2_adata,
            prefix='pvi'
        )

    kf_data = KFold(n_splits=4, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        mod1_train, mod1_test = mod1_adata[train_index].copy(), mod1_adata[test_index].copy()
        mod2_train, mod2_test = mod2_adata[train_index].copy(), mod2_adata[test_index].copy()
        gc.collect()

        model = ModelName(
            mod1_train,  # n_mod1_input
            mod2_train,  # n_mod2_input
            "rna", "atac",
            batch_col=batch_col,
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            counts_layer=trainer_params.get('counts_layer', None),
            use_decoder=model_params.get('use_decoder', False),
            emb_dim=model_params.get('emb_dim', 10),
            encoder_hidden_dims=model_params.get('hidden_dims', (128,)),
            decoder_hidden_dims=model_params.get('hidden_dims', (128)),
            seed=seed,
            variational=model_params.get('variational', True),
            combine_method=model_params.get('decode_method', 'dropout'),
            modality_discriminative=model_params.get('modality_discriminative', False),
            batch_discriminative=model_params.get('batch_discriminative', False),
            batch_dispersion=model_params.get('batch_dispersion', False),
            distance_loss=model_params.get('distance_loss', False),
            loss_method=model_params.get('loss_method', 'clip'),
            set_temperature=model_params.get('set_temperature', None),
            cap_temperature=model_params.get('cap_temperature', None),
            reconstruct_mod1_fn=reconstruct_mod1(scvi_model),
            reconstruct_mod2_fn=reconstruct_mod2(pvi_model)
        )

        model.train(
            epochs=trainer_params.get('n_epochs', 300),
            batch_size=trainer_params.get('batch_size', 5000),
            ckpt_dir=config['ckpt_dir']
        )

        sc._settings.ScanpyConfig.figdir = Path(model.trainer.ckpt_dir)

        train_latents = model.get_latent_representation()
        mod1_train.obsm['mod1_features'] = mod2_train.obsm['mod1_features'] = train_latents[0]
        mod1_train.obsm['mod2_features'] = mod2_train.obsm['mod2_features'] = train_latents[1]

        mod1_train.obsm['mod1_reconstruct'], _ = model.get_normalized_expression()
        mod1_train.obsm['mod1_cross'], _ = model.get_cross_modality_expression(mod2_train, mod1_to_mod2=False)

        test_latents = model.get_latent_representation(mod1_test, mod2_test)
        mod1_test.obsm['mod1_features'] = mod2_test.obsm['mod1_features'] = test_latents[0]
        mod1_test.obsm['mod2_features'] = mod2_test.obsm['mod2_features'] = test_latents[1]

        mod1_test.obsm['mod1_reconstruct'], _ = model.get_normalized_expression(mod1_test, mod2_test)
        mod1_test.obsm['mod1_cross'], _ = model.get_cross_modality_expression(mod2_test, mod1_to_mod2=False)

        save = {
            'mod1_features': mod1_train.obsm['mod1_features'],
            'mod2_features': mod1_train.obsm['mod2_features'],
            'mod1_reconstruct': mod1_train.obsm['mod1_reconstruct']
        }
        with open(os.path.join(model.trainer.ckpt_dir, 'train_embs.pkl'), 'wb') as f:
            pickle.dump(save, f)

        save = {
            'mod1_features': mod1_test.obsm['mod1_features'],
            'mod2_features': mod1_test.obsm['mod2_features'],
            'mod1_reconstruct': mod1_test.obsm['mod1_reconstruct']
        }
        with open(os.path.join(model.trainer.ckpt_dir, 'test_embs.pkl'), 'wb') as f:
            pickle.dump(save, f)

        # KNN evaluation
        for n in [5, 17, 29, 41, 53, 65]:
            train_X = np.concatenate((mod1_train.obsm['mod1_features'], mod1_train.obsm['mod2_features']), axis=1)
            train_y = mod1_train.obs[config['cell_type_col']]
            test_X = np.concatenate((mod1_test.obsm['mod1_features'], mod1_test.obsm['mod2_features']), axis=1)
            test_y = mod1_test.obs[config['cell_type_col']]
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(train_X, train_y)

            pred_y = neigh.predict(test_X)
            accuracy = np.sum(pred_y == test_y) / len(test_y)
            wandb.run.summary[f'Fold {i} Regular {n}-nn Average'] = accuracy
        
        gc.collect()
        # Assess reconstructions
        counts_layer = trainer_params.get('counts_layer', None)
        if isinstance(counts_layer, list):
            mod1_raw, _ = counts_layer[0], counts_layer[1]
        else:
            mod1_raw = counts_layer
        if mod1_raw is not None:
            # f.write(f'Mean-squared log error: {mean_squared_log_error(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
            wandb.run.summary[f'Fold {i} Pearson correlation'] = correlation_score(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])
        # else:
        #     f.write(f'Mean-squared log error: {mean_squared_log_error(mod1_test.X.toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
        #     f.write(f'Pearson correlation: {correlation_score(mod1_test.X.toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
        # gc.collect()

        # lls = []
        # for j in range(mod2_test.shape[0]):
        #     if mod2_raw is not None:
        #         x = mod2_test.layers[mod2_raw][j].toarray().flatten()
        #     else:
        #         x = mod2_test.X[j].toarray().flatten()
        #     y = mod2_test.obsm['mod2_reconstruct'][j]
        #     l = log_loss(x, y)
        #     lls.append(l)
        # f.write(f'ATAC BCE: {np.array(lls).mean()}\n')
        # gc.collect()

        # FOSCTTM
        res1 = foscttm(mod1_test.obsm['mod1_features'], mod1_test.obsm['mod2_features'])
        res2 = foscttm(mod1_test.obsm['mod2_features'], mod1_test.obsm['mod1_features'])
        wandb.run.summary[f'Fold {i} FOSCTTM 1'] = res1.mean()
        wandb.run.summary[f'Fold {i} FOSCTTM 2'] = res2.mean()

        gc.collect()

        # new_rna = ad.AnnData(mod1_test.obsm['mod1_reconstruct'].copy(), mod1_test.obs, mod1_test.var)
        # sc.pp.normalize_total(new_rna, target_sum=1e4)
        # sc.pp.log1p(new_rna)
        # sc.tl.pca(new_rna, svd_solver='arpack')
        # sc.pp.neighbors(new_rna)
        # sc.tl.umap(new_rna, min_dist=0.1)
        # sc.pl.umap(new_rna, color=[config['cell_type_col']], save=f'rna_reconstruction_{i}.png')
        # del new_rna
        # gc.collect()

        # # Embeddings of test data
        # concat = np.concatenate((mod1_test.obsm['mod1_features'], mod2_test.obsm['mod2_features']), axis=1)
        # mod1_test.obsm['concat'] = concat
        # sc.pp.neighbors(mod1_test, use_rep='mod2_features')
        # sc.tl.umap(mod1_test, min_dist=0.1)
        # sc.pl.umap(mod1_test, color=config['cell_type_col'], save=f'concat_clustering_{i}.png')
        # del mod1_test.obsm['concat']
        # gc.collect()

        # # Mixing of embeddings in test
        # concat_feat = np.concatenate([mod1_test.obsm['mod1_features'], mod2_test.obsm['mod2_features']])
        # concat = ad.concat([mod1_test, mod1_test], label='mod_feature')
        # concat.obsm['concat_feat'] = concat_feat
        # sc.pp.neighbors(concat, use_rep='concat_feat')
        # sc.tl.umap(concat, min_dist=0.5)
        # sc.pl.umap(concat, color=[config['cell_type_col'], 'mod_feature'], save=f'mix_test_embeddings_{i}.png')
        # del concat, concat_feat
        # gc.collect()

        # # Mixing of train and test embeddings
        # concat = ad.concat([mod1_train, mod1_test], label='train_test')
        # sc.pp.neighbors(concat, use_rep='mod1_features')
        # sc.tl.umap(concat, min_dist=0.5)
        # sc.pl.umap(concat, color=[config['cell_type_col'], 'train_test'], save=f'mix_train_test_{i}.png')
        # del concat
        # gc.collect()

        del mod1_train, mod1_test, mod2_train, mod2_test
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help='seed'
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config, args.seed)
