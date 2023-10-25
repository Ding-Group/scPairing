import os
import argparse

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import yaml
from pathlib import Path

import anndata as ad
import scanpy as sc

import scale
from scale import *
from scale.plot import *
from scale.utils import *


def main(config):
    files = config['files']
    model_params = config['model_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None

    _, mod2_files = files['mod1'], files['mod2']
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    mod2_adata.X = mod2_adata.layers['raw']

    res = SCALE_function(
        '/scratch/st-jiaruid-1/yinian/my_jupyter/mouse_brain_atac_scale.h5ad',
        n_feature=1,
        reference='celltype',
        batch_size=256,
        outdir='/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/SCALE/'
    )

    res.write('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/SCALE/mouse_brain_final.h5ad')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)