import os
import argparse
import sys
import gc

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

scvi.settings.seed = 5

def main(file1, file2, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    mod1_adata.var.loc[:, 'modality'] = 'Gene Expression'
    mod2_adata.var.loc[:, 'modality'] = 'Peaks'

    mod1_adata.X = mod1_adata.layers['counts'].copy()
    mod2_adata.X = mod2_adata.layers['binary'].copy()

    combined = ad.concat([mod1_adata, mod2_adata], axis=1, merge='first')
    del mod1_adata, mod2_adata
    gc.collect()

    n = 1
    adata_rna = combined[:n, combined.var.modality == "Gene Expression"].copy()
    adata_paired = combined[n : combined.n_obs-n].copy()
    adata_atac = combined[combined.n_obs - n :, combined.var.modality == "Peaks"].copy()

    del combined
    gc.collect()

    brain_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)
    sc.pp.filter_genes(brain_mvi, min_cells=int(brain_mvi.shape[0] * 0.01))

    brain_mvi = brain_mvi[:, brain_mvi.var["modality"].argsort()].copy()
    brain_mvi.X = brain_mvi.X.astype('int')
    scvi.model.MULTIVI.setup_anndata(brain_mvi, batch_key="modality", categorical_covariate_keys=['batch'])

    mvi = scvi.model.MULTIVI(
        brain_mvi,
        n_genes=(brain_mvi.var["modality"] == "Gene Expression").sum(),
        n_regions=(brain_mvi.var["modality"] == "Peaks").sum(),
    )
    mvi.view_anndata_setup()
    mvi.train()

    latent = mvi.get_latent_representation()
    with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}.pkl', 'wb') as f:
        pickle.dump(latent, f)
    
    mvi.save(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}_model')


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
