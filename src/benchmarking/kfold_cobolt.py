import os
import gc
import argparse

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import anndata as ad
import pandas as pd
import scanpy as sc
import muon as mu
import scipy

import yaml
from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import KFold

from cobolt.utils import SingleData, MultiomicDataset
from cobolt.model import Cobolt


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
    mod1_adata = ad.read_h5ad(file1)[:-1].copy()
    mod2_adata = ad.read_h5ad(file2)[:-1].copy()

    kf_data = KFold(n_splits=4, shuffle=True, random_state=2)
    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        rna_train, rna_test = mod1_adata[train_index].copy(), mod1_adata[test_index].copy()
        atac_train, atac_test = mod2_adata[train_index].copy(), mod2_adata[test_index].copy()

        rna_data = SingleData(
            "GeneExpr",
            "10X",
            rna_train.var.index.to_numpy(),
            rna_train.layers['counts'].toarray(),
            rna_train.obs.index.to_numpy()
        )
        atac_data = SingleData(
            "ChromAccess",
            "10X",
            atac_train.var.index.to_numpy(),
            atac_train.layers['counts'].toarray(),
            atac_train.obs.index.to_numpy()
        )
        rna_data2 = SingleData(
            "GeneExpr",
            "20X",
            rna_test.var.index.to_numpy(),
            rna_test.layers['counts'].toarray(),
            ('rna_' + rna_test.obs.index).to_numpy()
        )
        atac_data2 = SingleData(
            "ChromAccess",
            "20X",
            atac_test.var.index.to_numpy(),
            atac_test.layers['counts'].toarray(),
            ('atac_' + atac_test.obs.index).to_numpy()
        )

        rna_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
        atac_data.filter_features(upper_quantile=0.99, lower_quantile=0.7)
        rna_data2.filter_features(upper_quantile=0.99, lower_quantile=0.7)
        atac_data2.filter_features(upper_quantile=0.99, lower_quantile=0.7)

        atac_data.count = scipy.sparse.csr_matrix(atac_data.count)
        rna_data.count = scipy.sparse.csr_matrix(rna_data.count)
        atac_data2.count = scipy.sparse.csr_matrix(atac_data2.count)
        rna_data2.count = scipy.sparse.csr_matrix(rna_data2.count)

        multi_dt = MultiomicDataset.from_singledata(rna_data, atac_data, rna_data2, atac_data2)

        model = Cobolt(dataset=multi_dt, lr=1e-4, n_latent=10, batch_size=256)
        model.train(num_epochs=100)

        model.calc_all_latent()
        latent = model.get_all_latent()
        latent_raw = model.get_all_latent(correction=False)

        # mod1_adata.obsm['X_cobolt'] = latent[0]

        # sc._settings.ScanpyConfig.figdir = Path('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt')
        # sc.pp.neighbors(mod1_adata, use_rep='X_cobolt')
        # sc.tl.umap(mod1_adata)
        # sc.pl.umap(mod1_adata, color='cell_type', save=f'{output}_clustering.png')
        
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt/{output}_embs_{i}.pkl', 'wb') as f:
            pickle.dump({
                'latent': latent,
                'latent_raw': latent_raw
            }, f)

        res1 = foscttm(
            latent_raw[0][np.logical_and(pd.Index(latent_raw[1]).str.startswith('10X'), latent_raw[2] == 'GeneExpr')],
            latent_raw[0][np.logical_and(pd.Index(latent_raw[1]).str.startswith('10X'),  latent_raw[2] == 'ChromAccess')]
        )
        res2 = foscttm(
            latent_raw[0][np.logical_and(pd.Index(latent_raw[1]).str.startswith('10X'), latent_raw[2] == 'ChromAccess')],
            latent_raw[0][np.logical_and(pd.Index(latent_raw[1]).str.startswith('10X'),  latent_raw[2] == 'GeneExpr')]
        )
        res3 = foscttm(
            latent_raw[0][pd.Index(latent_raw[1]).str.startswith('20X~rna')],
            latent_raw[0][pd.Index(latent_raw[1]).str.startswith('20X~atac')]
        )
        res4 = foscttm(
            latent_raw[0][pd.Index(latent_raw[1]).str.startswith('20X~atac')],
            latent_raw[0][pd.Index(latent_raw[1]).str.startswith('20X~rna')]
        )
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cobolt/{output}_foscttm_{i}.txt', 'w') as f:
            f.write(f'{res1.mean()}, {res2.mean()}\n')
            f.write(f'{res3.mean()}, {res4.mean()}\n')
        
        del rna_train, rna_test, atac_train, atac_test
        gc.collect()


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
