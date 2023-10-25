import os
import argparse
import sys

sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scvi-tools/')

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import yaml
from pathlib import Path
import pickle

import anndata as ad
import scanpy as sc
import numpy as np
import scvi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold


scvi.settings.seed = 4


# def main(config, output_dir):
#     ref = sc.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_rna.h5ad')

#     ref = ref[:, ref.var.highly_variable].copy()
#     ref.X = ref.layers['counts']

#     scvi.model.SCVI.setup_anndata(ref, batch_key='batch')
#     model = scvi.model.SCVI(ref, n_latent=50)
#     model.train()
#     model.save(output_dir, overwrite=True, prefix='scvi_')

#     latent = model.get_latent_representation()
#     ref.obsm['X_scVI'] = latent

#     with open(os.path.join(output_dir, "embs.pkl"), 'wb') as f:
#         pickle.dump(latent, f)

#     X = ref.obsm['X_scVI']
#     y = ref.obs['cell_type']

#     kf = KFold(n_splits=10)
#     with open(os.path.join(output_dir, "knn.txt"), 'w') as f:
#         for n in [5, 17, 29, 41, 53, 65]:
#             vals = []
#             for i, (train_index, test_index) in enumerate(kf.split(X)):
#                 train_X, train_y = X[train_index], y[train_index]
#                 test_X, test_y = X[test_index], y[test_index]
#                 neigh = KNeighborsClassifier(n_neighbors=n)
#                 neigh.fit(train_X, train_y)

#                 pred_y = neigh.predict(test_X)
#                 vals.append(np.sum(pred_y == test_y) / len(test_y))
#             f.write(f'n={n}, Average={sum(vals) / len(vals)}\n')

#     # Visualization
#     sc._settings.ScanpyConfig.figdir = Path(output_dir)

#     sc.pp.neighbors(ref, use_rep='X_scVI')
#     sc.tl.umap(ref, min_dist=0.1)
#     sc.pl.umap(ref, color=['cell_type', 'batch'], ncols=1, save='vis.png')


def main(config, output_dir):
    ref = ad.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_merged.h5ad')
    # val = sc.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/validation_bmmc_merged.h5ad')

    # val.obs['batch'] = 'unknown'
    # val.obs['cell_type'] = 'unknown'

    # mod2_adata = ad.concat([ref, val], merge='first')
    mod2_adata = ref
    # mod2_adata.X = mod2_adata.layers['counts']
    mod2_adata.obs.batch = mod2_adata.obs.batch.cat.add_categories(['val'])

    # min_cells = int(mod2_adata.shape[0] * 0.01)
    # sc.pp.filter_genes(mod2_adata, min_cells=min_cells)

    scvi.model.PEAKVI.setup_anndata(mod2_adata, batch_key='batch')
    pvi = scvi.model.PEAKVI(mod2_adata, n_latent=50)
    pvi.train()

    pvi.save(output_dir, overwrite=True, prefix='pvi_merged_2_')

    latent = pvi.get_latent_representation()
    mod2_adata.obsm['X_PeakVI'] = latent

    with open(os.path.join(output_dir, "pvi_merged_2_embs.pkl"), 'wb') as f:
        pickle.dump(latent, f)
    
    # kNN
    # X = mod2_adata.obsm['X_PeakVI']
    # y = mod2_adata.obs[config['cell_type_col']]

    # kf = KFold(n_splits=10)
    # with open(os.path.join(output_dir, "knn.txt"), 'w') as f:
    #     for n in [5, 17, 29, 41, 53, 65]:
    #         vals = []
    #         for i, (train_index, test_index) in enumerate(kf.split(X)):
    #             train_X, train_y = X[train_index], y[train_index]
    #             test_X, test_y = X[test_index], y[test_index]
    #             neigh = KNeighborsClassifier(n_neighbors=n)
    #             neigh.fit(train_X, train_y)

    #             pred_y = neigh.predict(test_X)
    #             vals.append(np.sum(pred_y == test_y) / len(test_y))
    #         f.write(f'n={n}, Average={sum(vals) / len(vals)}\n')
    
    # Visualization
    # sc._settings.ScanpyConfig.figdir = Path(output_dir)

    # sc.pp.neighbors(mod2_adata, use_rep='X_PeakVI')
    # sc.tl.umap(mod2_adata, min_dist=0.1)
    # sc.pl.umap(mod2_adata, color=['cell_type', 'batch'], save='vis.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config, args.output)
