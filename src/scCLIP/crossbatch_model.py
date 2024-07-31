import sys
import os
import argparse
import gc
import pickle
sys.path.append('./')
# sys.path.append('/scratch/st-jiaruid-1/yinian/my_jupyter/scvi-tools/')

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113

import scanpy as sc
import numpy as np
import anndata as ad
import yaml
from pathlib import Path
import scipy
import torch
import wandb

from main import scPairing
from trainers.UnsupervisedTrainer import UnsupervisedTrainer
# import scvi

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, log_loss


def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient

    Source: https://www.kaggle.com/code/xiafire/lb-t15-msci-multiome-catboostregressor#Predicting
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


def foscttm(x: np.ndarray, y: np.ndarray, split=10000, **kwargs):
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    foscttms = []
    for i in range(0, x.shape[0], split):
        x_s = x[i: i + split]
        d = scipy.spatial.distance_matrix(x_s, y, **kwargs)
        foscttm_x = (d < np.expand_dims(np.diag(d, k=i), axis=1)).mean(axis=1)
        foscttms.append(foscttm_x)
    return np.concatenate(foscttms)


def main(config: dict, method: str, seed: int = 0):

    files = config['files']
    model_params = config['model_params']
    trainer_params = config['trainer_params']
    if config['cell_type_col'] == 'None':
        model_params['cell_type_col'] = None
    batch_col = trainer_params.get('batch_col', 'batch_indices')
    use_pretrained_decoder = config.get('pretrained', False)

    if method == 'scvi':
        m = 'X_scVI'
    elif method == 'scgpt':
        m = 'X_scGPT'
    elif method == 'cellplm':
        m = 'X_cellplm'
    else:
        raise ValueError("SOMETHING WRONG")
    config['method'] = method
    wandb.init(project="cross-batch", config=config)

    mod1_files, mod2_files = files['mod1'], files['mod2']
    mod1_adata = ad.concat([ad.read_h5ad(f) for f in mod1_files], label="batch_indices", merge='same')
    mod2_adata = ad.concat([ad.read_h5ad(f) for f in mod2_files], label="batch_indices", merge='same')

    # Forgot to add
    mod1_adata.obsm['X_pca'] = mod1_adata.obsm['X_pca'][:, :20]
    mod2_adata.obsm['X_lsi'] = mod2_adata.obsm['X_lsi'][:, :20]

    # New CellPLM
    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/cellplm/new_embs.pkl', 'rb') as f:
        mod1_adata.obsm['X_cellplm'] = pickle.load(f)

    del mod1_adata.uns, mod1_adata.obsp, mod2_adata.uns, mod2_adata.obsp

    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc_full_rna/embs.pkl', 'rb') as f:
        mod1_adata.obsm['X_scVI'] = pickle.load(f)

    # mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable].copy()
    sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.shape[0] * 0.01))

    # if use_pretrained_decoder:
    #     scvi_model = scvi.model.SCVI.load(
    #         '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc_full_rna/',
    #         # '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc/',
    #         adata=mod1_adata,
    #         prefix='scvi_'
    #     )
    #     pvi_model = scvi.model.PEAKVI.load(
    #         '/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/scVI/10x_bmmc',
    #         adata=mod2_adata,
    #         prefix='pvi'
    #     )

    for i, fold in enumerate(['s1', 's2', 's3', 's4']):
        mod1_adata_ref = mod1_adata[~mod1_adata.obs.batch.str.startswith(fold)].copy()
        mod1_adata_val = mod1_adata[mod1_adata.obs.batch.str.startswith(fold)].copy()
        mod2_adata_ref = mod2_adata[~mod2_adata.obs.batch.str.startswith(fold)].copy()
        mod2_adata_val = mod2_adata[mod2_adata.obs.batch.str.startswith(fold)].copy()

        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/notebooks/BenchmarkingNotebooks/CrossBatchResults/{method}_{fold}_held_out_ref_embs.pkl', 'rb') as f:
            mod1_adata_ref.obsm['X_scVI'] = pickle.load(f)
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/notebooks/BenchmarkingNotebooks/CrossBatchResults/{method}_{fold}_held_out_val_embs.pkl', 'rb') as f:
            mod1_adata_val.obsm['X_scVI'] = pickle.load(f)
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/notebooks/BenchmarkingNotebooks/CrossBatchResults/peakvi_{fold}_held_out_ref_embs.pkl', 'rb') as f:
            mod2_adata_ref.obsm['X_PeakVI'] = pickle.load(f)
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/notebooks/BenchmarkingNotebooks/CrossBatchResults/peakvi_{fold}_held_out_val_embs.pkl', 'rb') as f:
            mod2_adata_val.obsm['X_PeakVI'] = pickle.load(f)

        model = scPairing(
            mod1_adata_ref,  # n_mod1_input
            mod2_adata_ref,  # n_mod2_input
            "rna", "atac",
            batch_col=batch_col,
            transformed_obsm=[m, 'X_PeakVI'],
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
            # reconstruct_mod1_fn=reconstruct_mod1(scvi_model),
            # reconstruct_mod2_fn=reconstruct_mod2(pvi_model)
        )

        model.train(
            epochs=trainer_params.get('n_epochs', 300),
            batch_size=trainer_params.get('batch_size', 5000),
            # ckpt_dir=config['ckpt_dir']
            ckpt_dir=None
        )

        # sc._settings.ScanpyConfig.figdir = Path(model.trainer.ckpt_dir)

        train_latents = model.get_latent_representation()
        mod1_adata_ref.obsm['mod1_features'] = mod2_adata_ref.obsm['mod1_features'] = train_latents[0]
        mod1_adata_ref.obsm['mod2_features'] = mod2_adata_ref.obsm['mod2_features'] = train_latents[1]

        test_latents = model.get_latent_representation(mod1_adata_val, mod2_adata_val)
        mod1_adata_val.obsm['mod1_features'] = mod2_adata_val.obsm['mod1_features'] = test_latents[0]
        mod1_adata_val.obsm['mod2_features'] = mod2_adata_val.obsm['mod2_features'] = test_latents[1]

        # KNN evaluation
        for n in [5, 17, 29, 41, 53, 65]:
            train_X = np.concatenate((mod1_adata_ref.obsm['mod1_features'], mod1_adata_ref.obsm['mod2_features']), axis=1)
            train_y = mod1_adata_ref.obs[config['cell_type_col']]
            test_X = np.concatenate((mod1_adata_val.obsm['mod1_features'], mod1_adata_val.obsm['mod2_features']), axis=1)
            test_y = mod1_adata_val.obs[config['cell_type_col']]
            neigh = KNeighborsClassifier(n_neighbors=n)
            neigh.fit(train_X, train_y)

            pred_y = neigh.predict(test_X)
            accuracy = np.sum(pred_y == test_y) / len(test_y)
            wandb.run.summary[f'Fold {i} Regular {n}-nn Average'] = accuracy
        
        gc.collect()

        # FOSCTTM
        res1 = foscttm(mod1_adata_val.obsm['mod1_features'], mod1_adata_val.obsm['mod2_features'])
        res2 = foscttm(mod1_adata_val.obsm['mod2_features'], mod1_adata_val.obsm['mod1_features'])
        wandb.run.summary[f'Fold {i+1} FOSCTTM 1'] = res1.mean()
        wandb.run.summary[f'Fold {i+1} FOSCTTM 2'] = res2.mean()

        # del mod1_train, mod1_test, mod2_train, mod2_test
        gc.collect()
    
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input config path")
    parser.add_argument(
        "--path", type=str, required=True, help="Path of the experiment config"
    )
    parser.add_argument(
        "--method", type=str, required=True, help='Method'
    )
    parser.add_argument(
        "--seed", type=int, required=True, help='seed'
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.path).read_text())
    main(config, args.method, args.seed)
