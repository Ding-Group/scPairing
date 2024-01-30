import sys
sys.path.append('../src/scCLIP/')

import os
import gc

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import scipy
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

files = [
    '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/10x_bmmc/scCLIP_01_26-10_05_06/embs.pkl',
    '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/10x_bmmc/scCLIP_01_26-10_10_39/embs.pkl',
    '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/10x_bmmc/scCLIP_01_26-10_15_54/embs.pkl',
    '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/10x_bmmc/scCLIP_01_26-10_21_18/embs.pkl'
]


if __name__ == '__main__':
    adata = ad.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_rna.h5ad')
    # adata = ad.concat([adata[1: -1], adata[0], adata[-1]], axis=0)
    mod1_file = open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/knn.txt', 'w')
    # mod2_file = open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/10x_bmmc/reg_rnn_mod2.txt', 'w')
    for file in files:
        # adata = ad.read_h5ad(file)
        with open(file, 'rb') as f:
            d = pickle.load(f)
            adata.obsm['mod1_features'] = d['mod1_features']
            adata.obsm['mod2_features'] = d['mod2_features']
            adata.obsm['concat'] = np.concatenate([d['mod1_features'], d['mod2_features']], axis=1)

        for n in [5, 17, 29, 41, 53, 65]:
            vals = []
            for batch in adata.obs.batch.cat.categories:
                batch_adata = adata[adata.obs.batch == batch]
                other_adata = adata[adata.obs.batch != batch]
                train_X, train_y = other_adata.obsm['concat'], other_adata.obs['cell_type']
                test_X, test_y = batch_adata.obsm['concat'], batch_adata.obs['cell_type']

                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                vals.append(np.sum(pred_y == test_y) / len(test_y))
            mod1_file.write(f'Batch n={n}, Average={sum(vals) / len(vals)}\n')
            # print(f'n={n}, Average={sum(vals) / len(vals)}\n')
    #     del adata
    #     import gc; gc.collect()

        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        X = adata.obsm['concat']
        y = adata.obs['cell_type']
        for n in [5, 17, 29, 41, 53, 65]:
            vals = []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                train_X, train_y = X[train_index], y[train_index]
                test_X, test_y = X[test_index], y[test_index]
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                vals.append(np.sum(pred_y == test_y) / len(test_y))
            mod1_file.write(f'Regular n={n}, Average={sum(vals) / len(vals)}\n')

        # kf = KFold(n_splits=10, shuffle=True, random_state=0)
        # X = adata.obsm['mod2_features']
        # y = adata.obs['cell_type']
        # for n in [5, 17, 29, 41, 53, 65]:
        #     vals = []
        #     for i, (train_index, test_index) in enumerate(kf.split(X)):
        #         train_X, train_y = X[train_index], y[train_index]
        #         test_X, test_y = X[test_index], y[test_index]
        #         neigh = KNeighborsClassifier(n_neighbors=n)
        #         neigh.fit(train_X, train_y)

        #         pred_y = neigh.predict(test_X)
        #         vals.append(np.sum(pred_y == test_y) / len(test_y))
        #     mod2_file.write(f'n={n}, Average={sum(vals) / len(vals)}\n')
        
        # del adata
        gc.collect()

    mod1_file.close()
    # mod2_file.close()