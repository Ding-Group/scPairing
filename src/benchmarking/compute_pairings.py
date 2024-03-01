import os
os.environ['NUMBA_CACHE_DIR'] = '/scratch/st-jiaruid-1/yinian/tmp/' # https://github.com/scverse/scanpy/issues/2113
os.environ['MPLCONFIGDIR'] = "/scratch/st-jiaruid-1/yinian/tmp/"

import pickle

import numpy as np
import scipy

np.random.seed(0)

with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/retina/full_with_cosine_2/unpaired_mod1.pkl', 'rb') as f:
    mod1_embs = pickle.load(f)
with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/retina/full_with_cosine_2/unpaired_mod2.pkl', 'rb') as f:
    mod2_embs = pickle.load(f)

# Downsample
mask = np.sort(np.random.choice(mod1_embs.shape[0], size=mod1_embs.shape[0] // 2, replace=False))
mod1_embs = mod1_embs[mask]

dist_matrix = scipy.spatial.distance.cdist(
    mod1_embs, mod2_embs,
    metric='cosine'
)
print("Computed Distance Matrix")
dist_matrix = 1 - dist_matrix
dist_matrix[dist_matrix < 0.9] = 0  # Eliminate pairings with less than 0.9 similarity
print("Starting Linear Sum Assignment")
row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_matrix, maximize=True)

use = dist_matrix[row_ind, col_ind] != 0

with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/retina/full_with_cosine_2/mod1_indices.pkl', 'wb') as f:
    pickle.dump(mask, f)

with open('/scratch/st-jiaruid-1/yinian/my_jupyter/scCLIP/results/retina/full_with_cosine_2/pairing_cosine.pkl', 'wb') as f:
    pickle.dump({'indices': [row_ind, col_ind], 'use': use}, f)
