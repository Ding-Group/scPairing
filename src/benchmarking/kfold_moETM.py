import sys
sys.path.append("/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/libraries/moETM")
import os
os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
import argparse
import pickle
from pathlib import Path
import gc

import scanpy as sc
import anndata as ad
import scipy
import numpy as np
import pandas as pd
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

from moETM.train import Trainer_moETM, Train_moETM
from dataloader import load_nips_rna_atac_dataset, prepare_nips_dataset, data_process_moETM
from moETM.build_model import build_moETM


def main(file1, file2, seed, output):
    if not os.path.isdir(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}'):
        os.mkdir(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}')

    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    mod1_adata = mod1_adata[:, mod1_adata.var.highly_variable].copy()
    sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.n_obs * 0.01))

    batch_index = np.array(mod1_adata.obs['batch'].values)
    unique_batch = list(np.unique(batch_index))
    batch_index = np.array([unique_batch.index(xs) for xs in batch_index])

    obs = mod1_adata.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    obs = mod2_adata.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    mod1_adata.X = mod1_adata.layers['counts'].copy()
    mod2_adata.X = mod2_adata.layers['counts'].copy()

    n_total_sample = mod1_adata.shape[0]

    kf_data = KFold(n_splits=4, shuffle=True, random_state=seed)
    for i, (train_index, test_index) in enumerate(kf_data.split(mod1_adata)):
        if i >= 2:
            continue
        rna_train, rna_test = mod1_adata[train_index].copy(), mod1_adata[test_index].copy()
        atac_train, atac_test = mod2_adata[train_index].copy(), mod2_adata[test_index].copy()

        X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1 = data_process_moETM(rna_train, atac_train)
        X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1 = data_process_moETM(rna_test, atac_test)
        num_batch = len(batch_index_train_T.unique())
        input_dim_mod1 = X_mod1_train_T.shape[1]
        input_dim_mod2 = X_mod2_train_T.shape[1]
        train_num = X_mod1_train_T.shape[0]
        test_num = X_mod1_test_T.shape[0]

        num_topic = 20
        emd_dim = 400
        encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)

        encoder_mod1 = encoder_mod1.to('cuda')
        encoder_mod2 = encoder_mod2.to('cuda')
        decoder = decoder.to('cuda')

        trainer = Trainer_moETM(encoder_mod1, encoder_mod2, decoder, optimizer)

        Eval_kwargs = {}
        Eval_kwargs['batch_col'] = 'batch_indices'
        Eval_kwargs['plot_fname'] = 'moETM_delta'
        Eval_kwargs['cell_type_col'] = 'cell_type'
        Eval_kwargs['clustering_method'] = 'louvain'
        Eval_kwargs['resolutions'] = [1.0]
        Eval_kwargs['plot_dir'] = f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}'

        gc.collect()
        Total_epoch = 500
        batch_size = 2000
        Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
        Test_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1]
        Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs)

        trainer.encoder_mod1.to('cpu')
        trainer.encoder_mod2.to('cpu')

        res = trainer.get_embed(X_mod1_train_T, X_mod2_train_T)
        rna_train.obsm['X_moetm'] = res['delta']
        res = trainer.get_embed(X_mod1_test_T, X_mod2_test_T)
        rna_test.obsm['X_moetm'] = res['delta']

        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}/train_embs_{i}.pkl', 'wb') as f:
            pickle.dump(rna_train.obsm['X_moetm'], f)
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}/test_embs_{i}.pkl', 'wb') as f:
            pickle.dump(rna_test.obsm['X_moetm'], f)

        gc.collect()
        # kNN
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}/knn_{i}.txt', 'w') as f:
            for n in [5, 17, 29, 41, 53, 65]:
                train_X = rna_train.obsm['X_moetm']
                train_y = rna_train.obs['cell_type']
                test_X = rna_test.obsm['X_moetm']
                test_y = rna_test.obs['cell_type']
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                accuracy = np.sum(pred_y == test_y) / len(test_y)
                f.write(f'n={n}, Average={accuracy}\n')

        # # Visualization
        # sc._settings.ScanpyConfig.figdir = Path(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/moETM/{output}')

        # sc.pp.neighbors(rna_test, use_rep='X_moetm')
        # sc.tl.umap(mod1_adata, min_dist=0.1)
        # sc.pl.umap(mod1_adata, color=['cell_type', 'batch'], ncols=1, save=f'test_vis_{i}.png')
        gc.collect()


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
    parser.add_argument(
        "--seed", type=int, required=True, help="Seed"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    main(args.file1, args.file2, args.seed, args.output)
