import sys
import os
import argparse
import pickle
sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/src/scCLIP')

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import torch

from models.scCLIP import scCLIP
from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def main(config, seed=0):
    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None
    batch_col = trainer_params.get('batch_col', 'batch_indices')

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable].copy()
    sc.pp.filter_genes(mod2_adata, min_cells=mod2_adata.shape[0] * 0.01)

    if trainer_params.get('transformed_obsm', None):
        if isinstance(trainer_params['transformed_obsm'], str):
            trainer_params['transformed_obsm'] = [trainer_params['transformed_obsm'], trainer_params['transformed_obsm']]
        mod1_input = mod1_adata.obsm[trainer_params['transformed_obsm'][0]].shape[1]
        mod2_input = mod2_adata.obsm[trainer_params['transformed_obsm'][1]].shape[1]
    else:
        mod1_input = mod1_adata.n_vars
        mod2_input = mod2_adata.n_vars

        sc.pp.scale(mod1_adata, max_value=10)
        sc.pp.scale(mod2_adata, max_value=10)
    
    model = scCLIP(
        mod1_input,  # n_mod1_input
        mod2_input,  # n_mod2_input
        mod1_adata.n_vars,  # n_mod1_var
        mod2_adata.n_vars,  # n_mod2_var
        n_batches=mod1_adata.obs[batch_col].nunique(),
        mod2_type=model_params.get('mod2_type', 'atac'),
        emb_dim=model_params['emb_dim'],
        encoder_hidden_dims=model_params['hidden_dims'],
        decoder_hidden_dims=model_params['hidden_dims'],
        variational=model_params.get('variational', False),
        use_decoder=model_params['decode_features'],
        combine_method=model_params.get('decode_method', 'dropout'),
        modality_discriminative=model_params.get('modality_discriminative', False),
        batch_discriminative=model_params.get('batch_discriminative', False),
        batch_dispersion=model_params.get('batch_dispersion', False),
        distance_loss=model_params.get('distance_loss', False),
        loss_method=model_params.get('loss_method', 'clip'),
        tau=model_params.get('tau', 0.1),
        downsample_clip=model_params.get('downsample_clip', False),
        downsample_clip_prob=model_params.get('downsample_clip_prob', 0.5),
        set_temperature=model_params.get('set_temperature', None),
        cap_temperature=model_params.get('cap_temperature', None),
        seed=seed
    )

    trainer = UnsupervisedTrainer(
        model,
        mod1_adata,
        mod2_adata,
        counts_layer=trainer_params.get('counts_layer', None),
        transformed_obsm=trainer_params.get('transformed_obsm', None),
        weight_decay=trainer_params.get('weight_decay', 0),
        ckpt_dir=config['ckpt_dir'],
        batch_size=trainer_params['batch_size'],
    )

    with open(os.path.join(trainer.ckpt_dir, 'params.txt'), 'w') as f:
        f.write(str(config))

    trainer.train(
        n_epochs=trainer_params['n_epochs'],
        eval_every=trainer_params['eval_every'],
        batch_col=batch_col,
        need_reconstruction=model_params['decode_features'],
        ping_every=trainer_params.get('ping_every', None),
        eval_kwargs=dict(cell_type_col=config['cell_type_col']),
        n_samplers=1,
        save_model_ckpt=True,
        eval=False
    )

    emb_names = ['mod1_features', 'mod2_features']
    if model_params.get('decode_features', False) and config.get('reconstruct', False):
        emb_names += ['mod1_reconstruct']
    nll = model.get_cell_embeddings_and_nll(
        mod1_adata, mod2_adata, emb_names=emb_names,
        counts_layer=trainer_params.get('counts_layer', None),
        transformed_obsm=trainer_params.get('transformed_obsm', None),
        batch_size=1000, inplace=True
    )
    save = {
        'mod1_features': mod1_adata.obsm['mod1_features'],
        'mod2_features': mod1_adata.obsm['mod2_features'],
        'mod1_reconstruct': mod1_adata.obsm['mod1_reconstruct']
    }
    with open(os.path.join(trainer.ckpt_dir, 'embs.pkl'), 'wb') as f:
        pickle.dump(save, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help='seed'
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config, args.seed)
