import sys
sys.path.append("/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/libraries/MIRA/")
import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
import argparse
from pathlib import Path
import pickle

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

import mira


def main(file1, file2, seed, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.n_obs * 0.01))
    mod2_adata.X = mod2_adata.layers['counts'].copy()

    model = mira.topics.make_model(
        mod1_adata.n_obs, mod1_adata.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
        feature_type = 'expression',
        highly_variable_key='highly_variable',
        counts_layer='counts',
        categorical_covariates='batch',
        seed=seed
    )

    model.get_learning_rate_bounds(mod1_adata)

    model.set_learning_rates(1e-3, 0.1) # for larger datasets, the default of 1e-3, 0.1 usually works well.
    # model.plot_learning_rate_bounds(figsize=(7,3))

    topic_contributions = mira.topics.gradient_tune(model, mod1_adata)
    NUM_TOPICS = 10

    model = model.set_params(num_topics = NUM_TOPICS).fit(mod1_adata)

    model.predict(mod1_adata)

    model = mira.topics.make_model(
        *mod2_adata.shape,
        feature_type = 'accessibility',
        seed=seed,
    )

    model.get_learning_rate_bounds(mod2_adata)
    model.set_learning_rates(1e-3, 0.1)

    topic_contributions = mira.topics.gradient_tune(model, mod2_adata)
    NUM_TOPICS = 10
    model.set_params(num_topics = NUM_TOPICS).fit(mod2_adata)

    model.predict(mod2_adata)

    mod1_adata, mod2_adata = mira.utils.make_joint_representation(mod1_adata, mod2_adata)
    # sc.pp.neighbors(mod1_adata, use_rep = 'X_joint_umap_features', metric = 'manhattan',
            #    n_neighbors = 20)
    # sc.tl.umap(mod1_adata, min_dist = 0.1)

    if not os.path.isdir(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}'):
        os.mkdir(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}')
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}/joint_embs.pkl', 'wb') as f:
        pickle.dump(mod1_adata.obsm['X_joint_umap_features'], f)
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}/mod1_embs.pkl', 'wb') as f:
        pickle.dump(mod1_adata.obsm['X_umap_features'], f)
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}/mod2_embs.pkl', 'wb') as f:
        pickle.dump(mod2_adata.obsm['X_umap_features'], f)

    # kNN
    X = mod1_adata.obsm['X_joint_umap_features']
    y = mod1_adata.obs['cell_type']

    kf = KFold(n_splits=10)
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}/knn.txt', 'w') as f:
        for n in [5, 17, 29, 41, 53, 65]:
            vals = []
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                train_X, train_y = X[train_index], y[train_index]
                test_X, test_y = X[test_index], y[test_index]
                neigh = KNeighborsClassifier(n_neighbors=n, metric='manhattan')
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                vals.append(np.sum(pred_y == test_y) / len(test_y))
            f.write(f'n={n}, Average={sum(vals) / len(vals)}\n')

    # sc._settings.ScanpyConfig.figdir = Path(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/mira/{output}')

    # fig, ax = plt.subplots(1,1,figsize=(15,10))
    # sc.pl.umap(mod1_adata, color = 'cell_type', ax = ax, size = 20, title = '', save='vis.png')


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
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed"
    )
    args = parser.parse_args()
    np.random.seed(int(args.seed))
    main(args.file1, args.file2, int(args.seed), args.output)
