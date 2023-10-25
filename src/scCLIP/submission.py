import os

os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import gc
import pandas as pd
import pickle
from os.path import join
import numpy as np
from datetime import datetime
import csv
import scanpy as sc
import anndata as ad

multi_pred = sc.read_h5ad('/scratch/st-jiaruid-1/yinian/my_jupyter/multi_pred.h5ad')
cite_pred = sc.read_h5ad('/scratch/st-jiaruid-1/yinian/my_jupyter/cite_pred.h5ad')

eval_ids = pd.read_csv('/arc/project/st-jiaruid-1/yinian/multiome/evaluation_ids.csv')

f = open('/scratch/st-jiaruid-1/yinian/my_jupyter/submission.csv', 'w')
f.write('row_id,target\n')

isin1 = eval_ids.loc[:, 'cell_id'].isin(cite_pred.obs.index)
isin2 = eval_ids.loc[:, 'cell_id'].isin(multi_pred.obs.index)
for ind in eval_ids.index:
    row_id = eval_ids['row_id'][ind]
    cell_id = eval_ids['cell_id'][ind]
    gene_id = eval_ids['gene_id'][ind]
    if isin1[ind]:
        f.write(f'{row_id},{float(cite_pred[cell_id, gene_id].X)}\n')
    elif isin2[ind]:
        f.write(f'{row_id},{float(multi_pred[cell_id, gene_id].X)}\n')
    else:
        f.write(f'{row_id},0.0\n')

f.close()