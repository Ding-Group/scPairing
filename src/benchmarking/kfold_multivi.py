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

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

scvi.settings.seed = 0

def main(file1, file2, output):
    mod1_adata = ad.read_h5ad(file1)
    mod2_adata = ad.read_h5ad(file2)

    sc.pp.filter_genes(mod1_adata, min_cells=int(mod1_adata.shape[0] * 0.01))
    sc.pp.filter_genes(mod2_adata, min_cells=int(mod2_adata.shape[0] * 0.01))

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

    brain_mvi = brain_mvi[:, brain_mvi.var["modality"].argsort()].copy()
    brain_mvi.X = brain_mvi.X.astype('int')

    os.mkdir(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/')

    kf_data = KFold(n_splits=4, shuffle=True, random_state=4)
    for i, (train_index, test_index) in enumerate(kf_data.split(adata_paired)):
        # Keep the two unpaired RNA and ATAC cells in both the training and testing data.
        train_mvi, test_mvi = ad.concat([brain_mvi[train_index], brain_mvi[-2:]]), ad.concat([brain_mvi[test_index], brain_mvi[-2:]])
        scvi.model.MULTIVI.setup_anndata(train_mvi, batch_key="modality", categorical_covariate_keys=['batch'])
        scvi.model.MULTIVI.setup_anndata(test_mvi, batch_key="modality", categorical_covariate_keys=['batch'])

        gc.collect()

        mvi = scvi.model.MULTIVI(
            train_mvi,
            n_genes=(brain_mvi.var["modality"] == "Gene Expression").sum(),
            n_regions=(brain_mvi.var["modality"] == "Peaks").sum(),
        )
        mvi.view_anndata_setup()
        mvi.train()

        latent = mvi.get_latent_representation()
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/train_embs_{i}.pkl', 'wb') as f:
            pickle.dump(latent, f)
        train_mvi.obsm['X_mvi'] = latent
        
        latent = mvi.get_latent_representation(test_mvi)
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/test_embs_{i}.pkl', 'wb') as f:
            pickle.dump(latent, f)
        test_mvi.obsm['X_mvi'] = latent
        
        mvi.save(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/model_{i}')

        # Remove the two extraneous cells
        train_mvi = train_mvi[:-2]
        test_mvi = test_mvi[:-2]

        # kNN accuracies
        with open(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/knn_{i}.txt', 'w') as f:
            for n in [5, 17, 29, 41, 53, 65]:
                train_X = train_mvi.obsm['X_mvi']
                train_y = train_mvi.obs['cell_type']
                test_X = test_mvi.obsm['X_mvi']
                test_y = test_mvi.obs['cell_type']
                neigh = KNeighborsClassifier(n_neighbors=n)
                neigh.fit(train_X, train_y)

                pred_y = neigh.predict(test_X)
                accuracy = np.sum(pred_y == test_y) / len(test_y)
                f.write(f'n={n}, Average={accuracy}\n')
        gc.collect()

        sc._settings.ScanpyConfig.figdir = Path(f'/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/benchmarking/MultiVI/{output}/')

        # Embeddings of test data
        sc.pp.neighbors(test_mvi, use_rep='X_mvi')
        sc.tl.umap(test_mvi, min_dist=0.1)
        sc.pl.umap(test_mvi, color=['cell_type', 'batch'], save=f'test_clustering_{i}.png')

        gc.collect()

        # Mixing of training and test data
        concat = ad.concat([train_mvi, test_mvi], label='train_test')
        sc.pp.neighbors(concat, use_rep='X_mvi')
        sc.tl.umap(concat, min_dist=0.5)
        sc.pl.umap(concat, color=['cell_type', 'train_test'], save=f'train_test_mix_{i}.png')

        del concat
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
        "--output", type=str, required=True, help="Output directory"
    )
    args = parser.parse_args()

    main(args.file1, args.file2, args.output)
