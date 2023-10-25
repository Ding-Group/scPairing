import sys
import os
import argparse
import gc
sys.path.append('./')
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import scipy

from models.trueCLIP import scCLIP
from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer

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


def main(config):
    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    if model_params.get('decode_hvar_rna', False):
        mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable]
    else:
        mod1_adata.var.loc[:, 'highly_variable'] = True

    if model_params.get('decode_hvar_atac', False):
        mod2_adata = mod2_adata[:, mod2_adata.var.highly_variable]
    else:
        sc.pp.filter_genes(mod2_adata, min_cells=mod2_adata.shape[0] * 0.01)
        mod2_adata.var.loc[:, 'highly_variable'] = True

    if trainer_params.get('transformed_obsm', None):
        if isinstance(trainer_params['transformed_obsm'], str):
            trainer_params['transformed_obsm'] = [trainer_params['transformed_obsm'], trainer_params['transformed_obsm']]
        mod1_nvars = mod1_adata.obsm[trainer_params['transformed_obsm'][0]].shape[1]
        mod2_nvars = mod2_adata.obsm[trainer_params['transformed_obsm'][1]].shape[1]
    else:
        mod1_nvars = mod1_adata.n_vars
        mod2_nvars = mod2_adata.n_vars

    kf_data = KFold(n_splits=4, shuffle=True, random_state=0)

    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        mod1_train, mod1_test = mod1_adata[train_index].copy(), mod1_adata[test_index].copy()
        mod2_train, mod2_test = mod2_adata[train_index].copy(), mod2_adata[test_index].copy()
        gc.collect()

        model = scCLIP(
            mod1_nvars,
            mod2_nvars,
            np.sum(mod1_adata.var.highly_variable) if trainer_params['highly_variable'] else mod1_nvars,
            np.sum(mod2_adata.var.highly_variable) if trainer_params['highly_variable'] else mod2_nvars,
            n_batches=mod1_adata.obs.batch_indices.nunique(),
            mod2_type='atac',
            emb_dim=model_params['emb_dim'],
            hidden_dims=model_params['hidden_dims'],
            variational=model_params.get('variational', False),
            decode_features=model_params['decode_features'],
            encode_hvar=model_params.get('encode_hvar', False),
            decode_hvar=True,
            combine_method=model_params.get('decode_method', 'dropout'),
            dropout_in_eval=model_params.get('dropout_in_eval', True),
            discriminative=model_params.get('discriminative', False),
            distance_loss=model_params.get('distance_loss', False),
            loss_method=model_params.get('loss_method', 'clip'),
            tau=model_params.get('tau', 0.1),
            downsample_clip=model_params.get('downsample_clip', False),
            downsample_clip_prob=model_params.get('downsample_clip_prob', 0.5),
            set_temperature=model_params.get('set_temperature', None),
            cap_temperature=model_params.get('cap_temperature', None),
        )

        trainer = UnsupervisedTrainer(
            model,
            mod1_train,
            mod2_train,
            raw_layer=trainer_params['raw_layer'],
            transformed_layer=trainer_params['transformed_layer'],
            use_highly_variable=trainer_params['highly_variable'],
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            decode_highly_variable=trainer_params['decode_highly_variable'],
            weight_decay=trainer_params.get('weight_decay', 0),
            logit_weight_decay=trainer_params.get('logit_weight_decay', 0),
            ckpt_dir=config['ckpt_dir'],
            batch_size=trainer_params['batch_size'],
        )

        sc._settings.ScanpyConfig.figdir = Path(trainer.ckpt_dir)

        with open(os.path.join(trainer.ckpt_dir, 'params.txt'), 'w') as f:
            f.write(str(config))

        trainer.train(
            n_epochs=trainer_params['n_epochs'],
            eval_every=trainer_params['eval_every'],
            ping_every=trainer_params.get('ping_every', None),
            eval_kwargs=dict(cell_type_col=config['cell_type_col']),
            n_samplers=1,
            kl_warmup_ratio=trainer_params.get('kl_warmup_ratio', 0),
            max_kl_weight=trainer_params.get('max_kl_weight', 1),
            flip_clip_dist=trainer_params['flip_clip_dist'],
            flip_contrastive_reconstruct=trainer_params['flip_contrastive_reconstruct'],
            reconstruct_warmup_ratio=trainer_params.get('reconstruct_warmup_ratio', 0),
            reconstruct_cutoff_ratio=trainer_params.get('reconstruct_cutoff_ratio', 0),
            max_reconstruct_weight=trainer_params.get('max_reconstruct_weight', 1),
            save_model_ckpt=True,
            eval=False
        )

        emb_names = ['mod1_features', 'mod2_features']
        if model_params.get('decode_features', False) and config.get('reconstruct', False):
            emb_names += ['mod1_reconstruct', 'mod2_reconstruct']
        nll = model.get_cell_embeddings_and_nll(
            mod1_train, mod2_train, emb_names=emb_names,
            raw_layer=trainer_params.get('raw_layer', None),
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            batch_size=1000, inplace=True
        )
        nll = model.get_cell_embeddings_and_nll(
            mod1_test, mod2_test, emb_names=emb_names,
            raw_layer=trainer_params.get('raw_layer', None),
            transformed_obsm=trainer_params.get('transformed_obsm', None),
            batch_size=1000, inplace=True
        )
        mod1_test.write(os.path.join(trainer.ckpt_dir, f'test_fold_{i}.h5ad'))

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
            raw_layer = trainer_params.get('raw_layer', None)
            if isinstance(raw_layer, list):
                mod1_raw, mod2_raw = raw_layer[0], raw_layer[1]
            else:
                mod1_raw = mod2_raw = raw_layer
            if mod1_raw is not None:
                f.write(f'Mean-squared log error: {mean_squared_log_error(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
                f.write(f'Pearson correlation: {correlation_score(mod1_test.layers[mod1_raw].toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
            else:
                f.write(f'Mean-squared log error: {mean_squared_log_error(mod1_test.X.toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
                f.write(f'Pearson correlation: {correlation_score(mod1_test.X.toarray(), mod1_test.obsm["mod1_reconstruct"])}\n')
            gc.collect()

            lls = []
            for j in range(mod2_test.shape[0]):
                if mod2_raw is not None:
                    x = mod2_test.layers[mod2_raw][j].toarray().flatten()
                else:
                    x = mod2_test.X[j].toarray().flatten()
                y = mod2_test.obsm['mod2_reconstruct'][j]
                l = log_loss(x, y)
                lls.append(l)
            f.write(f'ATAC BCE: {np.array(lls).mean()}\n')
            gc.collect()

            # FOSCTTM
            res1 = foscttm(mod1_test.obsm['mod1_features'], mod1_test.obsm['mod2_features'])
            res2 = foscttm(mod1_test.obsm['mod2_features'], mod1_test.obsm['mod1_features'])
            f.write(f'FOSCTTM: {res1.mean()}, {res2.mean()}\n')
        gc.collect()

        new_rna = ad.AnnData(mod1_test.obsm['mod1_reconstruct'].copy(), mod1_test.obs, mod1_test.var)
        sc.pp.normalize_total(new_rna, target_sum=1e4)
        sc.pp.log1p(new_rna)
        sc.tl.pca(new_rna, svd_solver='arpack')
        sc.pp.neighbors(new_rna)
        sc.tl.umap(new_rna, min_dist=0.1)
        sc.pl.umap(new_rna, color=[config['cell_type_col']], save=f'rna_reconstruction_{i}.png')
        del new_rna
        gc.collect()

        # Embeddings of test data
        concat = np.concatenate((mod1_test.obsm['mod1_features'], mod2_test.obsm['mod2_features']), axis=1)
        mod1_test.obsm['concat'] = concat
        sc.pp.neighbors(mod1_test, use_rep='mod2_features')
        sc.tl.umap(mod1_test, min_dist=0.1)
        sc.pl.umap(mod1_test, color=config['cell_type_col'], save=f'concat_clustering_{i}.png')
        del mod1_test.obsm['concat']
        gc.collect()

        # Mixing of embeddings in test
        concat_feat = np.concatenate([mod1_test.obsm['mod1_features'], mod2_test.obsm['mod2_features']])
        concat = ad.concat([mod1_test, mod1_test], label='mod_feature')
        concat.obsm['concat_feat'] = concat_feat
        sc.pp.neighbors(concat, use_rep='concat_feat')
        sc.tl.umap(concat, min_dist=0.5)
        sc.pl.umap(concat, color=[config['cell_type_col'], 'mod_feature'], save=f'mix_test_embeddings_{i}.png')
        del concat, concat_feat
        gc.collect()

        # Mixing of train and test embeddings
        concat = ad.concat([mod1_train, mod1_test], label='train_test')
        sc.pp.neighbors(concat, use_rep='mod1_features')
        sc.tl.umap(concat, min_dist=0.5)
        sc.pl.umap(concat, color=[config['cell_type_col'], 'train_test'], save=f'mix_train_test_{i}.png')
        del concat
        gc.collect()

        del mod1_train, mod1_test, mod2_train, mod2_test
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)