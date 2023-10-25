import os

os.environ[ 'NUMBA_CACHE_DIR' ] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import scipy
import pickle

# def main(config, out):
#     ad2 = ad.read_h5ad(config)
#     ad2 = ad2[:, ad2.var.index.str.startswith('chr')].copy()

#     windows = epi.ct.make_windows(10000)
#     indices = []
#     chrom = []
#     start = []
#     end = []
#     for key in windows.keys():
#         ranges = windows[key]
#         for r in ranges:
#             indices.append(f'chr{key}:{r[0]}-{r[1]}')
#             chrom.append(key)
#             start.append(r[0])
#             end.append(r[1])
#     var = pd.DataFrame(index=indices, columns=['chrom', 'start', 'end'])
#     var.loc[:, 'chrom'] = var.index.str.split('[:-]').str[0]
#     var.loc[:, 'start'] = var.index.str.split('[:-]').str[1].astype('int')
#     var.loc[:, 'end'] = var.index.str.split('[:-]').str[2].astype('int')

#     ad2.var.loc[:, 'chrom'] = ad2.var.index.str.split('[:-]').str[0]
#     ad2.var.loc[:, 'start'] = ad2.var.index.str.split('[:-]').str[1].astype('int')
#     ad2.var.loc[:, 'end'] = ad2.var.index.str.split('[:-]').str[2].astype('int')

#     raw = scipy.sparse.csc_matrix(ad2.X)
#     data = scipy.sparse.lil_matrix((ad2.n_obs, len(indices)))

#     for j, ind in enumerate(var.index):
#         chrom = var.loc[ind]['chrom']
#         start = var.loc[ind]['start']
#         end = var.loc[ind]['end']
#         i = ((ad2.var['start'].between(start, end)) | ad2.var['end'].between(start, end)) & (ad2.var['chrom'] == chrom)
#         data[:, j] = raw[:, i].sum(1)
#         if j % 1000 == 0: print(j)

#     data = scipy.sparse.csr_matrix(data)
#     adata = ad.AnnData(data, obs=ad2.obs, var=var)

#     adata.write(out)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Input config path")
#     parser.add_argument(
#         "--path", type=str, required=True, help="Path of the AnnData"
#     )
#     parser.add_argument(
#         "--output", type=str, required=True, help="Path of output AnnData"
#     )
#     args = parser.parse_args()

#     main(args.path, args.output)

def main():
    adata1 = ad.read_h5ad("/arc/project/st-jiaruid-1/yinian/atac-rna/validation_bmmc_atac.h5ad")
    adata2 = ad.read_h5ad("/arc/project/st-jiaruid-1/yinian/atac-rna/10x_bmmc_atac.h5ad")
    
    adata2.var.loc[:, 'chrom'] = adata2.var.index.str.split('[:-]').str[0]
    adata2.var.loc[:, 'start'] = adata2.var.index.str.split('[:-]').str[1].astype('int')
    adata2.var.loc[:, 'end'] = adata2.var.index.str.split('[:-]').str[2].astype('int')

    ans = {}
    for index, row in adata1.var.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        i = (adata2.var['chrom'] == chrom) & \
            ((adata2.var['start'].between(start, end)) | \
            (adata2.var['end'].between(start, end)) | \
            (np.logical_and(adata2.var['start'] < start, adata2.var['end'] > end)))
        if len(adata2.var[i]) >= 1:
            ans[index] = adata2.var[i]

    data = scipy.sparse.lil_matrix((adata2.n_obs, len(ans.keys())))
    X = scipy.sparse.csc_matrix(adata2.layers['counts'])
    keys = list(ans.keys())

    for i, key in enumerate(keys):
        w = ans[key]
        data[:, i] += X[:, adata2.var.index.get_indexer(w.index)].sum(-1)
        if i % 1000 == 0:
            print(i)
    with open('/scratch/st-jiaruid-1/yinian/my_jupyter/matrix.pkl', 'wb') as f:
        pickle.dump(data, f)

    data = scipy.sparse.csr_matrix(data)
    new_adata2 = ad.AnnData(data, adata2.obs, pd.Index(keys))
    new_adata2.write("/scratch/st-jiaruid-1/yinian/my_jupyter/10x_bmmc_merged.h5ad")

    adata1[:, list(ans.keys())].write("/scratch/st-jiaruid-1/yinian/my_jupyter/validation_merged.h5ad")

main()