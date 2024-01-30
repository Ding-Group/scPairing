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

from main import ModelName
# from trainers.UnsupervisedTrainerCLIP import UnsupervisedTrainer

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

    model = ModelName(
        mod1_adata, mod2_adata,
        "rna", "atac",
        batch_col=batch_col,
        transformed_obsm=trainer_params.get('transformed_obsm', None),
        counts_layer=trainer_params.get('counts_layer', None),
        use_decoder=model_params.get('use_decoder', False),
        emb_dim=model_params.get('emb_dim', 10),
        encoder_hidden_dims=model_params.get('hidden_dims', (128,)),
        decoder_hidden_dims=model_params.get('hidden_dims', (128)),
        seed=seed,
        variational=model_params.get('variational', True),
        combine_method=model_params.get('decode_method', 'dropout'),
        modality_discriminative=model_params.get('modality_discriminative', False),
        batch_discriminative=model_params.get('batch_discriminative', False),
        batch_dispersion=model_params.get('batch_dispersion', False),
        distance_loss=model_params.get('distance_loss', False),
        loss_method=model_params.get('loss_method', 'clip'),
        set_temperature=model_params.get('set_temperature', None),
        cap_temperature=model_params.get('cap_temperature', None),
    )
    
    model.train(
        epochs=trainer_params.get('n_epochs', 300),
        batch_size=trainer_params.get('batch_size', 5000),
        ckpt_dir=config['ckpt_dir']
    )

    with open(os.path.join(model.trainer.ckpt_dir, 'params.txt'), 'w') as f:
        f.write(str(config))

    latents = model.get_latent_representation()
    mod1_adata.obsm['mod1_features'] = mod2_adata.obsm['mod1_features'] = latents[0]
    mod1_adata.obsm['mod2_features'] = mod2_adata.obsm['mod2_features'] = latents[1]

    if model_params.get('use_decoder', False) and config.get('reconstruct', False):
        mod1_adata.obsm['mod1_reconstruct'], _ = model.get_normalized_expression()

    save = {
        'mod1_features': mod1_adata.obsm['mod1_features'],
        'mod2_features': mod1_adata.obsm['mod2_features'],
        'mod1_reconstruct': mod1_adata.obsm['mod1_reconstruct']
    }

    with open(os.path.join(model.trainer.ckpt_dir, 'embs.pkl'), 'wb') as f:
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
