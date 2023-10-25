import sys
import os
import argparse
sys.path.append('./')
sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scvi-tools/')

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import torch
import scvi

from models.trueCLIP import scCLIP
from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def reconstruct_mod1(scvi_model):
    mse = torch.nn.MSELoss()
    def f(mod2_features, true_features, counts, library_size, cell_indices, is_training, batch_indices=None):
#         if is_training:
#             return None, mse(mod2_features, true_features)
        if batch_indices is None:
            batch_indices = torch.zeros(mod2_features.shape[0], device=mod2_features.device)
        library_size = torch.log(library_size)
        res = scvi_model.module.generative(mod2_features, library_size, batch_indices)
        loss = -res['px'].log_prob(counts).sum(-1).mean()
#         loss = mse(mod2_features, true_features)
        return res['px'].mu, loss
    return f

def reconstruct_mod2(pvi_model):
    mse = torch.nn.MSELoss()
    def f(mod1_features, true_features, counts, library_size, cell_indices, is_training, batch_indices=None):
#         if is_training:
#             return None, mse(mod1_features, true_features)
        if batch_indices is None:
            batch_indices = torch.zeros(mod1_features.shape[0], device=mod1_features.device)
        res = pvi_model.module.generative(mod1_features, mod1_features, batch_indices)
        dres = pvi_model.module.d_encoder(counts, batch_indices, ())
        region_factors = torch.sigmoid(pvi_model.module.region_factors[cell_indices]).reshape((mod1_features.shape[0], 1))
        loss = pvi_model.module.get_reconstruction_loss(res['p'], dres, region_factors, counts).mean()
#         loss = mse(mod1_features, true_features)
        return res['p'], loss
    return f


def main(config):
    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    scvi_params = config['scvi_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    if model_params.get('decode_hvar_rna', False):
        mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable].copy()
    else:
        mod1_adata.var.loc[:, 'highly_variable'] = True

    if model_params.get('decode_hvar_atac', False):
        mod2_adata = mod2_adata[:, mod2_adata.var.highly_variable].copy()
    else:
        sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.shape[0] * 0.01))
        mod2_adata.var.loc[:, 'highly_variable'] = True

    if trainer_params.get('transformed_obsm', None):
        if isinstance(trainer_params['transformed_obsm'], str):
            trainer_params['transformed_obsm'] = [trainer_params['transformed_obsm'], trainer_params['transformed_obsm']]
        mod1_nvars = mod1_adata.obsm[trainer_params['transformed_obsm'][0]].shape[1]
        mod2_nvars = mod2_adata.obsm[trainer_params['transformed_obsm'][1]].shape[1]
    else:
        mod1_nvars = mod1_adata.n_vars
        mod2_nvars = mod2_adata.n_vars

        sc.pp.scale(mod1_adata, max_value=10)
        sc.pp.scale(mod2_adata, max_value=10)

    scvi_model = scvi.model.SCVI.load(
        scvi_params['directory'],
        adata=mod1_adata,
        prefix=scvi_params['scvi_prefix']
    )
    pvi_model = scvi.model.PEAKVI.load(
        scvi_params['directory'],
        adata=mod2_adata,
        prefix=scvi_params['pvi_prefix']
    )

    model = scCLIP(
        mod1_nvars,
        mod2_nvars,
        np.sum(mod1_adata.var.highly_variable) if trainer_params['highly_variable'] else mod1_nvars,
        np.sum(mod2_adata.var.highly_variable) if trainer_params['highly_variable'] else mod2_nvars,
        n_batches=mod1_adata.obs.batch_indices.nunique(),
        reconstruct_mod1_fn=reconstruct_mod1(scvi_model),
        reconstruct_mod2_fn=reconstruct_mod2(pvi_model),
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
        mod1_adata,
        mod2_adata,
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
        save_model_ckpt=True
    )

    emb_names = ['mod1_features', 'mod2_features']
    if model_params.get('decode_features', False) and config.get('reconstruct', False):
        emb_names += ['mod1_reconstruct', 'mod2_reconstruct']
    nll = model.get_cell_embeddings_and_nll(
        mod1_adata, mod2_adata, emb_names=emb_names,
        raw_layer=trainer_params.get('raw_layer', None),
        transformed_obsm=trainer_params.get('transformed_obsm', None),
        batch_size=1000, inplace=True
    )
    mod1_adata.write(os.path.join(trainer.ckpt_dir, 'final.h5ad'))

    with open(os.path.join(trainer.ckpt_dir, 'knn.txt'), 'w') as f:
        X = mod1_adata.obsm['mod1_features']
        y = mod1_adata.obs[config['cell_type_col']]

        kf = KFold(n_splits=10)

        for n in [5, 17, 29, 41, 53, 65]:
            vals = []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                train_X, train_y = X[train_index], y[train_index]
                test_X, test_y = X[test_index], y[test_index]
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                vals.append(np.sum(pred_y == test_y) / len(test_y))
            f.write(f'n={n}, Average={sum(vals) / len(vals)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)