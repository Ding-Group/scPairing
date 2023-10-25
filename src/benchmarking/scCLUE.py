import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import argparse
import yaml
from pathlib import Path
import gc
import numpy as np

import scanpy as sc
import anndata as ad
import muon as mu
from muon import atac as ac
import scglue
import pickle
import scipy

def foscttm(x: np.ndarray, y: np.ndarray, **kwargs):
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
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return foscttm_x, foscttm_y


def main(file1, file2, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    rna = mod1_adata
    atac = mod2_adata
    
    rna.uns['log1p'] = {'base': None}
    sc.pp.highly_variable_genes(rna, min_mean=0.05, max_mean=1.5, min_disp=.5)
    sc.pp.filter_genes(atac, min_cells=int(atac.n_obs * 0.01))
    mod2_adata.var['highly_variable'] = True

    scglue.models.configure_dataset(
        mod1_adata, "NB", use_highly_variable=True,
        use_rep="X_pca", use_layer='raw', use_obs_names=True
    )

    scglue.models.configure_dataset(
        mod2_adata, "NB", use_highly_variable=True,
        use_rep="X_lsi", use_layer='raw', use_obs_names=True
    )

    ActiveModel = scglue.models.SCCLUEModel

    glue = ActiveModel(
        {"rna": rna, "atac": atac},
        latent_dim=10, random_seed=4
    )
    glue.compile()
    gc.collect()
    glue.fit(
        {"rna": rna, "atac": atac},
        batch_size=500,
        align_burnin=2,
        max_epochs=500,
        patience=3,
        directory=f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/clue/{output}/'
    )

    rna.obsm["X_glue"] = glue.encode_data("rna", rna)
    atac.obsm["X_glue"] = glue.encode_data("atac", atac)

    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/clue/{output}/embs_5.pkl', 'wb') as f:
        pickle.dump({"mod1": rna.obsm['X_glue'], "mod2": atac.obsm['X_glue']}, f)

    res = foscttm(rna.obsm['X_glue'], atac.obsm['X_glue'])
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/clue/{output}/foscttm_5.txt', 'w') as f:
        f.write(f'{res[0].mean()}, {res[1].mean()}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--file1", type=str, required=True, help="Path of the first AnnData"
    )
    parser.add_argument(
        "--file2", type=str, required=True, help="Path of the second AnnData"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output file name"
    )
    args = parser.parse_args()

    main(args.file1, args.file2, args.output)
