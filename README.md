# scPairing

This model borrows code for model training and batch sampling from [scETM](https://github.com/hui2000ji/scETM) (Zhao *et al.*, 2021) and terminology from [scVI](https://github.com/scverse/scvi-tools) (Lopez *et al.*, 2018).
We thank the respective authors for making their code available to the community.

## Installation

Dependencies:

* python>=3.6
* torch>=1.5
* numpy>=1.16.2
* matplotlib>=3.1.2
* scikit-learn>=0.20.3
* h5py>=2.9.0
* pandas>=0.25
* tqdm>=4.31.1
* anndata>=0.7
* scanpy>=1.4.6
* scipy>=1.0
* pandas>=1.5.0

Optional dependencies required by the tutorials:
* scvi-tools>=1.0.0
* harmonypy
* muon

### Install from Git

To install scPairing, run `pip install git+https://github.com/Ding-Group/scPairing.git`.
If you wish to run the tutorials, there are additional dependencies, which can be installed alongside scPairing by running `pip install "scPairing[tutorials]@git+https://github.com/Ding-Group/scPairing.git"`.

## Usage

In the `tutorials/` directory, we provide four examples of scPairing usage, two on joint scRNA-seq and scATAC-seq data, and two on trimodal data.

### scPairing model

The `scPairing` class contains the model and methods to train the model and compute embeddings.
Full documentation of `scPairing` initialization and methods can be found in the `docs`.

### triscPairing model

The `trisciPairing` class contains the extension of scPairing to trimodal data.
The usage is nearly identical with a few modifications to account for the extra modality.
Full documentation of `triscPairing` can be found in the `docs`.

### Artificial pairing procedure

Given a paired multiomics bridge dataset and two unpaired unimodal datasets, the following procedure can generate a pairing between the two unimodal datasets to create artificial multiomics data:

1. Create and train a scPairing model on the multiomics bridge dataset.
    ```python
    model = scPairing(...) # See docs/ or tutorials/ for scPairing parameters information
    model.train()
    ```
2. Project the unimodal datasets onto the common embedding space.
    ```python
    _, latents1 = model.get_cross_modality_expression(True, mod1_adata_query)
    _, latents2 = model.get_cross_modality_expression(False, mod2_adata_query)
    ```
    This produces `latents1` and `latents2`, which are the embeddings for the two unimodal datasets on the common embedding space, respectively.
3. Compute a maximum weight bipartite matching, where the graph edge weights are given by the cosine similarity between cells from each modality.
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.optimize import linear_sum_assignment

    CUTOFF = 0.8

    sim_matrix = cosine_similarity(latents1, latents2)  # Pairwise similarity matrix
    sim_matrix[sim_matrix < CUTOFF] = 0  # Do not allow pairings between cells with similarity less than CUTOFF

    # mod1_ind and mod2_ind store the corresponds, where mod1_ind[0] corresponds with mod2_ind[0], and so on
    mod1_ind, mod2_ind = linear_sum_assignment(sim_matrix, maximize=True)
    # Since linear_sum_assignment will pair cells with similarity 0 towards the end, filter those pairings out
    use = sim_matrix[mod1_ind, mod2_ind] != 0
    ```
4. Re-order the data by the pairings
    ```python
    paired_mod1_adata = mod1_adata_query[mod1_ind[use]]
    paired_mod2_adata = mod2_adata_query[mod2_ind[use]]
    ```

For an example, see `tutorials/Re-pairing data with scPairing.ipynb`.

## References

R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef. Deep generative modeling for singlecell
transcriptomics. Nature Methods, 15(12):1053–1058, 2018.

Y. Zhao, H. Cai, Z. Zhang, J. Tang, and Y. Li. Learning interpretable cellular and gene signature
embeddings from single-cell transcriptomic data. Nature Communications, 12:5261, 2021.