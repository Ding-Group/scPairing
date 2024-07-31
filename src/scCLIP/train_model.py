import sys
import os
import argparse
import pickle
import wandb
import json
sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/src/scCLIP')

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import anndata as ad
import numpy as np
import yaml
import scipy
from pathlib import Path

from main import scPairing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import scib

def foscttm(x: np.ndarray, y: np.ndarray, split=10000, **kwargs):
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    foscttm_x, foscttm_y
        FOSCTTM for samples in modality X and Y, respectively

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    foscttms = []
    for i in range(0, x.shape[0], split):
        x_s = x[i: i + split]
        d = scipy.spatial.distance_matrix(x_s, y, **kwargs)
        foscttm_x = (d < np.expand_dims(np.diag(d, k=i), axis=1)).mean(axis=1)
        foscttms.append(foscttm_x)
    return np.concatenate(foscttms)


def main(config, seed=0, rep1=None, rep2=None):
    config['seed'] = seed

    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None
    batch_col = trainer_params.get('batch_col', 'batch_indices')
    transformed_obsm = [rep1, rep2] if rep1 is not None else trainer_params.get('transformed_obsm', None)
    config.update({'transformed_obsm': transformed_obsm})
    wandb.init(project="new-benchmarking", config=config)
    

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    del mod1_adata.uns, mod1_adata.obsp, mod2_adata.uns, mod2_adata.obsp

    # Forgot to add
    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc_full_rna/embs.pkl', 'rb') as f:
        mod1_adata.obsm['X_scVI'] = pickle.load(f)

    # New CellPLM
    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cellplm/new_embs.pkl', 'rb') as f:
        mod1_adata.obsm['X_cellplm'] = pickle.load(f)

    mod1_adata.obsm['X_pca'] = mod1_adata.obsm['X_pca'][:, :20]
    mod2_adata.obsm['X_lsi'] = mod2_adata.obsm['X_lsi'][:, :20]

    model = scPairing(
        mod1_adata, mod2_adata,
        "rna", "atac",
        batch_col=batch_col,
        transformed_obsm=transformed_obsm,
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
    )
    
    model.train(
        epochs=trainer_params.get('n_epochs', 300),
        batch_size=trainer_params.get('batch_size', 5000),
        # ckpt_dir=config['ckpt_dir']
        ckpt_dir=None
    )

    latents = model.get_latent_representation()
    mod1_adata.obsm['mod1_features'] = mod2_adata.obsm['mod1_features'] = latents[0]
    mod1_adata.obsm['mod2_features'] = mod2_adata.obsm['mod2_features'] = latents[1]
    mod1_adata.obsm['concat'] = np.concatenate(latents, axis=1)

    # save = {
    #     'mod1_features': mod1_adata.obsm['mod1_features'],
    #     'mod2_features': mod1_adata.obsm['mod2_features'],
    # }

    # if model_params.get('use_decoder', False) and config.get('reconstruct', False):
    #     mod1_adata.obsm['mod1_reconstruct'], _ = model.get_normalized_expression()
    #     save['mod1_reconstruct'] = mod1_adata.obsm['mod1_reconstruct']

    # with open(os.path.join(model.trainer.ckpt_dir, 'embs.pkl'), 'wb') as f:
    #     pickle.dump(save, f)

    for n in [5, 17, 29, 41, 53, 65]:
        vals = []
        for batch in mod1_adata.obs.batch.cat.categories:
            batch_adata = mod1_adata[mod1_adata.obs.batch == batch]
            other_adata = mod1_adata[mod1_adata.obs.batch != batch]
            train_X, train_y = other_adata.obsm['concat'], other_adata.obs['cell_type']
            test_X, test_y = batch_adata.obsm['concat'], batch_adata.obs['cell_type']

            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(train_X, train_y)

            pred_y = neigh.predict(test_X)
            vals.append(np.sum(pred_y == test_y) / len(test_y))
        wandb.run.summary[f'Batch {n}-nn Average'] = sum(vals) / len(vals)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    X = mod1_adata.obsm['concat']
    y = mod1_adata.obs['cell_type']
    for n in [5, 17, 29, 41, 53, 65]:
        vals = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(train_X, train_y)

            pred_y = neigh.predict(test_X)
            vals.append(np.sum(pred_y == test_y) / len(test_y))
        wandb.run.summary[f'Regular {n}-nn Average'] = sum(vals) / len(vals)

    res = foscttm(mod1_adata.obsm['mod1_features'], mod2_adata.obsm['mod2_features'])
    wandb.run.summary['foscttm1'] = res.mean()
    res = foscttm(mod2_adata.obsm['mod2_features'], mod1_adata.obsm['mod1_features'])
    wandb.run.summary['foscttm2'] = res.mean()

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help='seed'
    )
    parser.add_argument(
        "--rep1", type=str, help="mod1 representation"
    )
    parser.add_argument(
        "--rep2", type=str, help="mod2 representation"
    )
    args = parser.parse_args()
    

    config = yaml.safe_load(Path(args.path).read_text())
    main(config, args.seed, args.rep1, args.rep2)
