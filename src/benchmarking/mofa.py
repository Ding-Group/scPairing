import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import argparse
import yaml
from pathlib import Path
import pickle

import numpy as np

import scanpy as sc
import anndata as ad
import muon as mu
from muon import atac as ac


def main(config):
    files = config['files']
    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod1_files], label="batch_indices", merge="same")
    mod2_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod2_files], label="batch_indices", merge="same")

    rna = mod1_adata
    rna = rna[:, rna.var.highly_variable].copy()
    atac = mod2_adata

    sc.pp.scale(rna, max_value=10)
    sc.pp.filter_genes(atac, min_cells=int(atac.shape[0] * 0.02))
    sc.pp.scale(atac, max_value=10)

    mdata = mu.MuData({"mod1": rna, "mod2": atac})
    mu.tl.mofa(
        mdata,
        groups_label="mod1:batch",
        n_factors=20,
        use_var=None,
        seed=5,
        outfile='/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MOFA/10x_bmmc_batch_5.hdf5'
    )
    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MOFA/10x_bmmc_batch_X_5.pkl', 'wb') as f:
        pickle.dump(mdata.obsm['X_mofa'], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)