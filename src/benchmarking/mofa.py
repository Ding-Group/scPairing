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
from muon import atac as ac
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def main(config):
    files = config['files']
    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod1_files], label="batch_indices")
    mod2_adata = ad.concat([ad.read_h5ad(r_file) for r_file in mod2_files], label="batch_indices")

    rna = mod1_adata
    atac = mod2_adata
    rna.X = rna.layers['raw'].copy()
    atac.X = atac.layers['raw'].copy()

    # Preprocessing on RNA data
    sc.pp.calculate_qc_metrics(rna, percent_top=None, log1p=False, inplace=True)
    mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
    mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)

    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)

    rna.raw = rna

    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, svd_solver='arpack')

    # Preprocessing on ATAC data
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)

    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 10)
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= 2000) & (x <= 15000))
    mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= 4000) & (x <= 40000))

    ac.pp.tfidf(atac, scale_factor=1e4)
    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)

    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)

    atac.raw = atac

    mdata = mu.MuData({"mod1": mod1_adata, "mod2": mod2_adata})
    mu.tl.mofa(
        mdata,
        n_factors=15,
        verbose=True,
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