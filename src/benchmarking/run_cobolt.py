import os
import argparse

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import anndata as ad
import pandas as pd
import scanpy as sc
import muon as mu
import scipy
import torch

import yaml
from pathlib import Path
import numpy as np
import pickle

from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt

torch.manual_seed(3)
np.random.seed(3)

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


def main(file1, file2, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    rna_data = SingleData(
        "GeneExpr",
        "10X",
        mod1_adata.var.index.to_numpy(),
        mod1_adata.layers['raw'].toarray(),
        mod1_adata.obs.index.to_numpy()
    )

    atac_data = SingleData(
        "ChromAccess",
        "10X",
        mod2_adata.var.index.to_numpy(),
        mod2_adata.layers['raw'].toarray(),
        mod2_adata.obs.index.to_numpy()
    )

    rna_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
    atac_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)

    atac_data.count = scipy.sparse.csr_matrix(atac_data.count)
    rna_data.count = scipy.sparse.csr_matrix(rna_data.count)

    multi_dt = MultiomicDataset.from_singledata(rna_data, atac_data)

    model = Cobolt(dataset=multi_dt, lr=1e-4, n_latent=10)
    model.train(num_epochs=100)

    model.calc_all_latent()
    latent = model.get_all_latent()
    latent_raw = model.get_all_latent(correction=False)
    
    mod1_adata.obsm['X_cobolt'] = latent[0]

    sc._settings.ScanpyConfig.figdir = Path('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt')
    sc.pp.neighbors(mod1_adata, use_rep='X_cobolt')
    sc.tl.umap(mod1_adata)
    sc.pl.umap(mod1_adata, color='cell_type', save=f'{output}_clustering.png')
    
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt/{output}_embs_3.pkl', 'wb') as f:
        pickle.dump({
            'latent': latent,
            'latent_raw': latent_raw
        }, f)

    res1 = foscttm(latent_raw[0][latent_raw[2] == 'GeneExpr'], latent_raw[0][latent_raw[2] == 'ChromAccess'])
    res2 = foscttm(latent_raw[0][latent_raw[2] == 'ChromAccess'], latent_raw[0][latent_raw[2] == 'GeneExpr'])
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt/{output}_foscttm_3.txt', 'w') as f:
        f.write(f'{res1.mean()}, {res2.mean()}\n')


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
