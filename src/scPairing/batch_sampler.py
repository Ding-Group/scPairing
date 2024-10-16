from typing import Iterator, List, Mapping, Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch
import torch.sparse
from scipy.sparse import spmatrix
from torch import FloatTensor


class CellSampler():
    """An iterable cell dataset for minibatch sampling. Assumes that each AnnData object has the
    same cells in the same order.

    Parameters
    ----------
    adata1
        AnnData object corresponding to the first modality of a multimodal single-cell dataset.
    adata2
        AnnData object corresponding to the second modality of a multimodal single-cell dataset.
    batch_size
        Minibatch size.
    counts_layer
        Key(s) in ``adata1.layers`` and ``adata2.layers`` corresponding to the raw counts for each modality.
        If a string is provided, the same key will be applied to both ``adata1.layers`` and ``adata2.layers``.
        If ``None`` is provided, raw counts will be taken from ``adata1.X`` and/or ``adata2.X``.
    transformed_obsm
        Key(s) in ``adata1.obsm`` and ``adata2.obsm`` corresponding to the low-dimension
        representations of each individual modality. If a string is provided, the same key will
        be applied to both ``adata1.obsm`` and ``adata2.obsm``.
    sample_batch_id
        Whether to yield batch indices in each sample.
    require_counts
        Whether the raw counts are required by the model. The raw counts are not needed
        if the model does not require a decoder.
    n_epochs
        Number of epochs to sample before raising StopIteration.
    rng
        The random number generator. Could be None if shuffle is False.
    batch_col
        A key in every AnnData's ``.obs`` for the batch column. Only used when sample_batch_id is True.
    shuffle
        whether to shuffle the dataset at the beginning of each epoch. qQ
        When n_cells >= batch_size, this attribute is ignored.
    """
    def __init__(self,
        adata1: anndata.AnnData,
        adata2: Optional[anndata.AnnData] = None,
        adata3: Optional[anndata.AnnData] = None,
        batch_size: int = 2000,
        counts_layer: List[Union[str, None]] = [None, None],
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        sample_batch_id: bool = False,
        require_counts: bool = True,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        batch_col: str = 'batch_indices',
        shuffle: bool = True
    ) -> None:
        self.n_cells: int = adata1.n_obs
        self.batch_size: int = batch_size
        self.n_epochs: Union[int, float] = n_epochs
        self.require_counts: bool = require_counts

        if isinstance(counts_layer, str) or counts_layer is None:
            counts_layer = [counts_layer, counts_layer, counts_layer]
        if transformed_obsm is None or isinstance(transformed_obsm, str):
                transformed_obsm = [transformed_obsm, transformed_obsm, transformed_obsm]

        self.has_mod2: bool = adata2 is not None
        self.has_mod3: bool = adata3 is not None

        self.X_1: Union[np.ndarray, spmatrix] = adata1.layers[counts_layer[0]] if counts_layer[0] else adata1.X
        self.is_sparse_1: bool = isinstance(self.X_1, spmatrix)
        self.library_size_1: Union[spmatrix, np.ndarray] = self.X_1.sum(1) if self.is_sparse_1 else self.X_1.sum(1, keepdims=True)
        self.X_1_transformed: Union[np.ndarray, spmatrix] = adata1.obsm[transformed_obsm[0]] if transformed_obsm[0] else adata1.X

        if adata2 is not None:
            self.X_2: Union[np.ndarray, spmatrix] = adata2.layers[counts_layer[1]] if counts_layer[1] else adata2.X        
            self.is_sparse_2: bool = isinstance(self.X_2, spmatrix)
            self.library_size_2: Union[spmatrix, np.ndarray] = self.X_2.sum(1) if self.is_sparse_2 else self.X_2.sum(1, keepdims=True)
            self.X_2_transformed: Union[spmatrix, np.ndarray] = adata2.obsm[transformed_obsm[1]] if transformed_obsm[1] else adata2.X
        if adata3 is not None:
            self.X_3: Union[np.ndarray, spmatrix] = adata3.layers[counts_layer[2]] if counts_layer[2] else adata3.X
            self.is_sparse_3: bool = isinstance(self.X_3, spmatrix)
            self.library_size_3: Union[spmatrix, np.ndarray] = self.X_3.sum(1) if self.is_sparse_3 else self.X_3.sum(1, keepdims=True)
            self.X_3_transformed: Union[spmatrix, np.ndarray] = adata3.obsm[transformed_obsm[2]] if transformed_obsm[2] else adata3.X

        if shuffle:
            self.rng: Union[None, np.random.Generator] = rng or np.random.default_rng()
        else:
            self.rng: Union[None, np.random.Generator] = None
        self.shuffle: bool = shuffle

        self.sample_batch_id: bool = sample_batch_id
        if self.sample_batch_id:
            assert batch_col in adata1.obs, f'{batch_col} not in adata.obs'
            self.batch_indices: pd.Series = adata1.obs[batch_col].astype('category').cat.codes
            
    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """Creates an iterator.

        If self.n_cells <= self.batch_size, simply returns the whole batch for
        self.n_epochs times.
        Otherwise, randomly or sequentially (depending on self.shuffle) sample
        minibatches of size self.batch_size.

        Yields:
            A dict mapping tensor names to tensors. The returned tensors
            include (B for batch_size, G for # modality 1, P for # modality 2):
                * X_1: the cell-first_modality matrix of shape [B, G].
                * X_2: the cell-second_modality matrix of shape [B, P]
                * library_size_1: total #modality 1 features for each cell [B].
                * library_size_2: total #modality 2 features for each cell [B].
                * cell_indices: the cell indices in the original dataset [B].
                * batch_indices (optional): the batch indices of each cell [B].
                    Is only returned if self.sample_batch_id is True.
        """
        if self.batch_size < self.n_cells:
            return self._low_batch_size()
        else:
            return self._high_batch_size()

    def _high_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """The iterator for the high batch size case.

        Simply returns the whole batch for self.n_epochs times.
        """
        count = 0

        result_dict = dict()
        if not self.require_counts:
            X_1 = torch.zeros([])
        else:
            X_1 = FloatTensor(self.X_1.todense()) if isinstance(self.X_1, spmatrix) else FloatTensor(self.X_1)
        library_size_1 = FloatTensor(self.library_size_1)
        X_1_transformed = FloatTensor(self.X_1_transformed.todense()) if isinstance(self.X_1_transformed, spmatrix) else FloatTensor(self.X_1_transformed)
        result_dict['cells_1'] = X_1
        result_dict['library_size_1'] = library_size_1
        result_dict['cells_1_transformed'] = X_1_transformed

        if self.has_mod2:
            if not self.require_counts:
                X_2 = torch.zeros([])
            else:
                X_2 = FloatTensor(self.X_2.todense()) if  isinstance(self.X_2, spmatrix) else FloatTensor(self.X_2)
            library_size_2 = FloatTensor(self.library_size_2)
            X_2_transformed = FloatTensor(self.X_2_transformed.todense()) if isinstance(self.X_2_transformed, spmatrix) else FloatTensor(self.X_2_transformed)
            result_dict['cells_2'] = X_2
            result_dict['library_size_2'] = library_size_2
            result_dict['cells_2_transformed'] = X_2_transformed

        if self.has_mod3:
            if not self.require_counts:
                X_3 = torch.zeros([])
            else:
                X_3 = FloatTensor(self.X_3.todense()) if  isinstance(self.X_3, spmatrix) else FloatTensor(self.X_3)
            library_size_3 = FloatTensor(self.library_size_3)
            X_3_transformed = FloatTensor(self.X_3_transformed.todense()) if isinstance(self.X_3_transformed, spmatrix) else FloatTensor(self.X_3_transformed)
            result_dict['cells_3'] = X_3
            result_dict['library_size_3'] = library_size_3
            result_dict['cells_3_transformed'] = X_3_transformed

        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)
        result_dict['cell_indices'] = cell_indices
        if self.sample_batch_id:
            result_dict['batch_indices'] = torch.LongTensor(self.batch_indices)
        while count < self.n_epochs:
            count += 1
            yield result_dict

    def _low_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:
        """The iterator for the low batch size case.

        Randomly or sequentially (depending on self.shuffle) sample minibatches
        of size self.batch_size.
        """
        entry_index = 0
        count = 0
        cell_range = np.arange(self.n_cells)
        if self.shuffle:
            self.rng.shuffle(cell_range)
        while count < self.n_epochs:
            if entry_index + self.batch_size >= self.n_cells:
                count += 1
                batch = cell_range[entry_index:]
                if self.shuffle:
                    self.rng.shuffle(cell_range)
                excess = entry_index + self.batch_size - self.n_cells
                if excess > 0 and count < self.n_epochs:
                    batch = np.append(batch, cell_range[:excess], axis=0)
                    entry_index = excess
                else:
                    entry_index = 0
            else:
                batch = cell_range[entry_index: entry_index + self.batch_size]
                entry_index += self.batch_size

            result_dict = dict()
            library_size_1 = FloatTensor(self.library_size_1[batch])
            result_dict['library_size_1'] = library_size_1
            if self.has_mod2:
                library_size_2 = FloatTensor(self.library_size_2[batch])
                result_dict['library_size_2'] = library_size_2
            if self.has_mod3:
                result_dict['library_size_3'] = FloatTensor(self.library_size_3[batch])

            if not self.require_counts:
                result_dict['cells_1'] = result_dict['cells_2'] = result_dict['cells_3'] = torch.zeros([])
            else:
                X_1 = self.X_1[batch]
                result_dict['cells_1'] = FloatTensor(X_1.todense()) if isinstance(X_1, spmatrix) else FloatTensor(X_1)
                if self.has_mod2:
                    X_2 = self.X_2[batch]
                    result_dict['cells_2'] = FloatTensor(X_2.todense()) if isinstance(X_2, spmatrix) else FloatTensor(X_2)
                if self.has_mod3:
                    X_3 = self.X_3[batch]
                    result_dict['cells_3'] = FloatTensor(X_3.todense()) if isinstance(X_3, spmatrix) else FloatTensor(X_3)

            X_1_transformed = self.X_1_transformed[batch]
            result_dict['cells_1_transformed'] = FloatTensor(X_1_transformed.todense()) if isinstance(X_1_transformed, spmatrix) else FloatTensor(X_1_transformed)
            if self.has_mod2:
                X_2_transformed = self.X_2_transformed[batch]
                result_dict['cells_2_transformed'] = FloatTensor(X_2_transformed.todense()) if isinstance(X_2_transformed, spmatrix) else FloatTensor(X_2_transformed)
            if self.has_mod3:
                X_3_transformed = self.X_3_transformed[batch]
                result_dict['cells_3_transformed'] = FloatTensor(X_3_transformed.todense()) if isinstance(X_3_transformed, spmatrix) else FloatTensor(X_3_transformed)

            cell_indices = torch.LongTensor(batch)
            result_dict['cell_indices'] = cell_indices
            if self.sample_batch_id:
                result_dict['batch_indices'] = torch.LongTensor(self.batch_indices[batch])
            yield result_dict
