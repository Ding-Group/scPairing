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

from models.scCLIP import scCLIP
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

    if trainer_params.get('transformed_obsm', None):
        if isinstance(trainer_params['transformed_obsm'], str):
            trainer_params['transformed_obsm'] = [trainer_params['transformed_obsm'], trainer_params['transformed_obsm']]
        mod1_input = mod1_adata.obsm[trainer_params['transformed_obsm'][0]].shape[1]
        mod2_input = mod2_adata.obsm[trainer_params['transformed_obsm'][1]].shape[1]
    else:
        mod1_input = mod1_adata.n_vars
        mod2_input = mod2_adata.n_vars

        sc.pp.scale(mod1_adata, max_value=10)
        sc.pp.scale(mod2_adata, max_value=10)

    kf_data = KFold(n_splits=4, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        mod1_train, mod1_test = mod1_adata[train_index].copy(), mod1_adata[test_index].copy()
        mod2_train, mod2_test = mod2_adata[train_index].copy(), mod2_adata[test_index].copy()
        gc.collect()

        model = scCLIP(
            mod1_input,  # n_mod1_input
            mod2_input,  # n_mod2_input
            mod1_adata.n_vars,  # n_mod1_var
            mod2_adata.n_vars,  # n_mod2_var
            n_batches=mod1_adata.obs[batch_col].nunique(),
            mod2_type=model_params.get('mod2_type', 'atac'),
            emb_dim=model_params['emb_dim'],
            reconstruct_mod1_fn=reconstruct_mod1(scvi_model) if use_pretrained_decoder else None,
            reconstruct_mod2_fn=reconstruct_mod2(pvi_model) if use_pretrained_decoder else None,
            encoder_hidden_dims=model_params['hidden_dims'],
            decoder_hidden_dims=model_params['hidden_dims'],
            variational=model_params.get('variational', False),
            use_decoder=model_params['use_decoder'],
            use_norm=model_params.get('norm', 'batch'),
            combine_method=model_params.get('decode_method', 'dropout'),
            modality_discriminative=model_params.get('modality_discriminative', False),
            batch_discriminative=model_params.get('batch_discriminative', False),
            batch_dispersion=model_params.get('batch_dispersion', False),
            distance_loss=model_params.get('distance_loss', False),
            loss_method=model_params.get('loss_method', 'clip'),
            tau=model_params.get('tau', 0.1),
            downsample_clip=model_params.get('downsample_clip', False),
            downsample_clip_prob=model_params.get('downsample_clip_prob', 0.5),
            set_temperature=model_params.get('set_temperature', None),
            cap_temperature=model_params.get('cap_temperature', None),
            seed=seed
        )

        trainer = UnsupervisedTrainer(
            model,
            mod1_train,
            mod2_train,
            counts_layer=trainer_params.get('counts_layer', None),
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            weight_decay=trainer_params.get('weight_decay', 0),
            ckpt_dir=config['ckpt_dir'],
            batch_size=trainer_params['batch_size'],
        )

        sc._settings.ScanpyConfig.figdir = Path(trainer.ckpt_dir)

        with open(os.path.join(trainer.ckpt_dir, 'params.txt'), 'w') as f:
            f.write(str(config))

        trainer.train(
            n_epochs=trainer_params['n_epochs'],
            eval_every=trainer_params['eval_every'],
            batch_col=batch_col,
            need_reconstruction=not use_pretrained_decoder,
            ping_every=trainer_params.get('ping_every', None),
            eval_kwargs=dict(cell_type_col=config['cell_type_col']),
            n_samplers=1,
            save_model_ckpt=True,
            eval=False
        )

        emb_names = ['mod1_features', 'mod2_features']
        if model_params.get('use_decoder', False) and config.get('reconstruct', False):
            emb_names += ['mod1_reconstruct']
        nll = model.get_cell_embeddings_and_nll(
            mod1_train, mod2_train, emb_names=emb_names,
            counts_layer=trainer_params.get('counts_layer', None),
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            batch_size=1000, inplace=True
        )
        nll = model.get_cell_embeddings_and_nll(
            mod1_test, mod2_test, emb_names=emb_names,
            counts_layer=trainer_params.get('counts_layer', None),
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            batch_size=1000, inplace=True
        )

        save = {
            'mod1_features': mod1_train.obsm['mod1_features'],
            'mod2_features': mod1_train.obsm['mod2_features'],
            'mod1_reconstruct': mod1_train.obsm['mod1_reconstruct']
        }
        with open(os.path.join(trainer.ckpt_dir, 'train_embs.pkl'), 'wb') as f:
            pickle.dump(save, f)

        save = {
            'mod1_features': mod1_test.obsm['mod1_features'],
            'mod2_features': mod1_test.obsm['mod2_features'],
            'mod1_reconstruct': mod1_test.obsm['mod1_reconstruct']
        }
        with open(os.path.join(trainer.ckpt_dir, 'test_embs.pkl'), 'wb') as f:
            pickle.dump(save, f)

        # KNN evaluation
        with open(os.path.join(trainer.ckpt_dir, f'knn_{i}.txt'), 'w') as f:
            for n in [5, 17, 29, 41, 53, 65]:
                train_X = np.concatenate((mod1_train.obsm['mod1_features'], mod1_train.obsm['mod2_features']), axis=1)
                train_y = mod1_train.obs[config['cell_type_col']]
                test_X = np.concatenate((mod1_test.obsm['mod1_features'], mod1_test.obsm['mod2_features']), axis=1)
                test_y = mod1_test.obs[config['cell_type_col']]
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                accuracy = np.sum(pred_y == test_y) / len(test_y)
                f.write(f'n={n}, Average={accuracy}\n')
        
        gc.collect()
        # Assess reconstructions
        with open(os.path.join(trainer.ckpt_dir, f'reconstructions_{i}.txt'), 'w') as f:
            # counts_layer = trainer_params.get('counts_layer', None)
            # if isinstance(counts_layer, list):
            #     mod1_raw, mod2_raw = counts_layer[0], counts_layer[1]
            # else:
            #     mod1_raw = mod2_raw = counts_layer
            # if mod1_raw is not None:
            #     f.write(f'Mean-squared log error: {mean_squared_log_error(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
            #     f.write(f'Pearson correlation: {correlation_score(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
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
            f.write(f'FOSCTTM: {res1.mean()}, {res2.mean()}\n')
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
