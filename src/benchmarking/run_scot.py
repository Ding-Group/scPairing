import os
import argparse

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import sys
sys.path.insert(1, '/arc/project/st-jiaruid-1/yinian/SCOT/src/')

import yaml
from pathlib import Path
import pickle

import anndata as ad
import scanpy as sc
import muon as mu
import numpy as np
import scipy
from sklearn.model_selection import KFold

from scotv1 import SCOT

def main(file1, file2, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    kf_data = KFold(n_splits=4, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        if i == 0:
            rna_test = mod1_adata[test_index].copy()
            atac_test = mod2_adata[test_index].copy()


    obsm1, obsm2 = 'X_scVI', 'X_PeakVI'
    scot_aligner = SCOT(rna_test.obsm[obsm1], atac_test.obsm[obsm2])
    k = 50 # a hyperparameter of the model, determines the number of neighbors to be used in the kNN graph constructed for cells based on sequencing data correlations
    e = 1e-3 # another hyperparameter of the model, determines the coefficient of the entropic regularization term
    normalize = True #
    aligned_domain1, aligned_domain2 = scot_aligner.align(k=k, e=e, normalize=normalize)

    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/SCOT/{output}', 'wb') as f:
        pickle.dump({
            'aligned_domain1': aligned_domain1,
            'aligned_domain2': aligned_domain2,
            'scot_aligner': scot_aligner
        }, f)


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
