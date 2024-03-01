import os
import argparse
import sys

sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scvi-tools/')

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import yaml
from pathlib import Path
import pickle
import gc

import anndata as ad
import scanpy as sc
# import numpy as np
import scvi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# def main(output_dir, seed):
#     scvi.settings.seed = seed
#     ref = sc.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_rna.h5ad')

#     ref.X = ref.layers['counts']

#     kf_data = KFold(n_splits=4, shuffle=True, random_state=0)
#     for i, (train_index, test_index) in enumerate(kf_data.split(ref)):
#         ref_train = ref[train_index].copy()

#         scvi.model.SCVI.setup_anndata(ref_train, batch_key='batch')
#         model = scvi.model.SCVI(ref_train, n_latent=50, dispersion='gene-batch', gene_likelihood='nb')
#         model.train()
#         model.save(output_dir, overwrite=True, prefix=f'scvi_fold{i}_')

#         latent = model.get_latent_representation()
#         ref_train.obsm['X_scVI'] = latent

#         with open(os.path.join(output_dir, f"scvi_embs_fold{i}.pkl"), 'wb') as f:
#             pickle.dump(latent, f)
#         gc.collect()


def main(output_dir, seed):
    scvi.settings.seed = seed
    ref = ad.read_h5ad('/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_atac.h5ad')

    min_cells = int(ref.shape[0] * 0.01)
    sc.pp.filter_genes(ref, min_cells=min_cells)

    ref.X = ref.layers['counts']
    kf_data = KFold(n_splits=4, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf_data.split(ref)):
        ref_train = ref[train_index].copy()

        scvi.model.PEAKVI.setup_anndata(ref_train, batch_key='batch')
        model = scvi.model.PEAKVI(ref_train, n_latent=50)
        model.train()
        model.save(output_dir, overwrite=True, prefix=f'pvi_fold{i}_')

        latent = model.get_latent_representation()
        ref_train.obsm['X_PeakVI'] = latent

        with open(os.path.join(output_dir, f"pvi_embs_fold{i}.pkl"), 'wb') as f:
            pickle.dump(latent, f)
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed"
    )

    args = parser.parse_args()

    main(args.output, args.seed)
