import sys
import os
import argparse
sys.path.append('./')
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import torch
import yaml
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
# import umap

from models.scCLIP import scCLIP, ClipLoss
from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer
from eval_utils import evaluate


def main(config):
    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices")
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices")

    model = scCLIP(
        mod1_adata.n_vars,
        mod2_adata.n_vars,
        mod1_adata.obs.batch_indices.nunique(),
        emb_dim=model_params['emb_dim'],
        hidden_dims=model_params['hidden_dims'],
        decode_features=model_params['decode_features'],
        decode_method=model_params['decode_method'],
        cell_dropout=model_params['cell_dropout']
    )

    trainer = UnsupervisedTrainer(
        model,
        mod1_adata,
        mod2_adata,
        raw_layer=trainer_params['raw_layer'],
        transformed_layer=trainer_params['transformed_layer'],
        ckpt_dir=config['ckpt_dir'],
        batch_size=trainer_params['batch_size'],
    )

    trainer.train(
        n_epochs=trainer_params['n_epochs'],
        eval_every=trainer_params['eval_every'],
        eval_kwargs=dict(cell_type_col=config['cell_type_col']),
        n_samplers=1,
        flip_contrastive_reconstruct=trainer_params['flip_contrastive_reconstruct'],
        reconstruct_warmup_ratio=trainer_params['reconstruct_warmup_ratio'],
        reconstruct_cutoff_ratio=trainer_params['reconstruct_cutoff_ratio'],
        max_reconstruct_weight=trainer_params['max_reconstruct_weight'],
        save_model_ckpt=True
    )

    nll = model.get_cell_embeddings_and_nll(mod1_adata, mod2_adata, emb_names = ['mod1_features', 'mod2_features'], batch_size=100, inplace=True)
    mod1_adata.write_csvs(trainer.ckpt_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)