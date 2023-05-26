import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import argparse
import yaml
from pathlib import Path

import numpy as np
import scipy

import scanpy as sc
import anndata as ad
import muon as mu
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def main(config):
    files = config['files']
    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod1_files], label="batch_indices")
    mod2_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod2_files], label="batch_indices")

    # Use the highly-variable genes/peaks
    # mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable]
    # mod2_adata = mod2_adata[:, mod2_adata.var.highly_variable]

    mdata = mu.MuData({"mod1": mod1_adata, "mod2": mod2_adata})
    mu.tl.mofa(
        mdata,
        use_var='highly_variable',
        outfile='/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MOFA/model.hdf5'
    )
    mdata.write_h5mu('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MOFA/mdata_mofa.mu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)