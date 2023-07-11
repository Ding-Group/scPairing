import sys
import os
import argparse
sys.path.append('./')
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
# import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
# import umap

# from models.scCLIP import scCLIP
from models.scCLIP_share_vi import scCLIP
from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer


def main(config):
    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    if model_params.get('decode_hvar', False):
        mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable]
        mod2_adata = mod2_adata[:, mod2_adata.var.highly_variable]

    sc.pp.scale(mod1_adata, max_value=10)
    sc.pp.scale(mod2_adata, max_value=10)

    model = scCLIP(
        mod1_adata.n_vars,
        mod2_adata.n_vars,
        mod1_adata.obs.batch_indices.nunique(),
        n_mod1_var=np.sum(mod1_adata.var.highly_variable) if trainer_params['highly_variable'] else None,
        n_mod2_var=np.sum(mod2_adata.var.highly_variable) if trainer_params['highly_variable'] else None,
        emb_dim=model_params['emb_dim'],
        hidden_dims=model_params['hidden_dims'],
        decode_features=model_params['decode_features'],
        decode_hvar=model_params['decode_hvar'],
        combine_method=model_params['decode_method'],
        loss_method=model_params['loss_method'],
        cell_dropout=model_params['cell_dropout'],
        discriminative=model_params['discriminative'],
        downsample_clip=model_params.get('downsample_clip', False),
        downsample_clip_prob=model_params.get('downsample_clip_prob', 0.5),
        set_temperature=model_params.get('set_temperature', None),
        cap_temperature=model_params.get('cap_temperature', None),
        loss_ratio=model_params['loss_ratio']
    )

    trainer = UnsupervisedTrainer(
        model,
        mod1_adata,
        mod2_adata,
        raw_layer=trainer_params['raw_layer'],
        transformed_layer=trainer_params['transformed_layer'],
        use_highly_variable=trainer_params['highly_variable'],
        decode_highly_variable=trainer_params['decode_highly_variable'],
        weight_decay=trainer_params.get('weight_decay', 0),
        logit_weight_decay=trainer_params.get('logit_weight_decay', 0),
        ckpt_dir=config['ckpt_dir'],
        batch_size=trainer_params['batch_size'],
    )

    with open(os.path.join(trainer.ckpt_dir, 'params.txt'), 'w') as f:
        f.write(str(config))

    trainer.train(
        n_epochs=trainer_params['n_epochs'],
        eval_every=trainer_params['eval_every'],
        ping_every=trainer_params.get('ping_every', None),
        eval_kwargs=dict(cell_type_col=config['cell_type_col']),
        n_samplers=1,
        flip_clip_dist=trainer_params['flip_clip_dist'],
        flip_contrastive_reconstruct=trainer_params['flip_contrastive_reconstruct'],
        reconstruct_warmup_ratio=trainer_params['reconstruct_warmup_ratio'],
        reconstruct_cutoff_ratio=trainer_params['reconstruct_cutoff_ratio'],
        max_reconstruct_weight=trainer_params['max_reconstruct_weight'],
        save_model_ckpt=True
    )

    emb_names = ['mod1_features', 'mod2_features']
    if config['reconstruct']:
        emb_names += ['mod1_reconstruct', 'mod2_reconstruct']
    nll = model.get_cell_embeddings_and_nll(mod1_adata, mod2_adata, emb_names=emb_names, batch_size=250, inplace=True)
    mod1_adata.write(os.path.join(trainer.ckpt_dir, 'final.h5ad'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config)