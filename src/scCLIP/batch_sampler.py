import threading
from typing import Any, Iterator, List, Mapping, Union, Callable, Optional

import anndata
import numpy as np
import pandas as pd
import torch
import torch.sparse
from scipy.sparse import spmatrix


class CellSampler():
    """An iterable cell dataset for minibatch sampling.

    Attributes:
        n_cells: number of cells in the dataset.
        batch_size: size of each sampled minibatch.
        n_epochs: number of epochs to sample before raising StopIteration.
        X_1: a (dense or sparse) matrix containing the cell-first_modality matrix.
        X_2: a (dense or sparse) matrix containing the cell-second_modality matrix.
        X_1_transformed: X_1 after applying the first modality transformation.
        X_2_transformed: X_2 after applying the second modality transformation.
        is_sparse_1: whether self.X_1 is a sparse matrix.
        is_sparse_2: whether self.X_2 is a sparse matrix
        shuffle: whether to shuffle the dataset at the beginning of each epoch.
            When n_cells >= batch_size, this attribute is ignored.
        rng: the random number generator. Could be None if shuffle is False.
        library_size_1: a (dense or sparse) vector storing the first modality library size for
            each cell.
        library_size_2: a (dense or sparse) vector storing the second modality library
            size for each cell.
        sample_batch_id: whether to yield batch indices in each sample.
        batch_indices: a (dense or sparse) vector storing the batch indices
            for each cell. Only present if sample_batch_id is True.
    """

    def __init__(self,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        batch_size: int,
        counts_layer: List[Union[str, None]] = [None, None],
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        sample_batch_id: bool = False,
        require_counts: bool = True,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        batch_col: str = 'batch_indices',
        shuffle: bool = True
    ) -> None:
        """Initializes the CellSampler object.

        Args:
            adata_1: an AnnData object storing the dataset of the first modality.
            adata_2: an AnnData object storing the dataset of the second modality.
            batch_size: size of each sampled minibatch.
            counts_layer: AnnData layer corresponding to raw counts. If it is a singular str,
                the same counts_layer will be applied to both AnnDatas. If it is a list,
                the first will be applied to adata_1, second applied to adata_2
            transformed_obsm: AnnData obsm key corresponding to transformed data.
            sample_batch_id: whether to yield batch indices in each sample.
            require_counts: Whether the raw counts are required by the model. The raw counts are not needed
                if the model is not going to decode anything
            n_epochs: number of epochs to sample before raising StopIteration.
            rng: the random number generator.
                Could be None if shuffle is False.
            batch_col: a key in adata.obs to the batch column. Only used when
                sample_batch_id is True.
            shuffle: whether to shuffle the dataset at the beginning of each
                epoch. When n_cells >= batch_size, this attribute is ignored.
        
        Assumption that adata_1 and adata_2 have the same cells in the same order.
        """
        assert adata_1.n_obs == adata_2.n_obs, "The two AnnData objects have a different number of observations"
        self.n_cells: int = adata_1.n_obs
        self.batch_size: int = batch_size
        self.n_epochs: Union[int, float] = n_epochs
        self.require_counts: bool = require_counts

        if isinstance(counts_layer, str) or counts_layer is None:
            counts_layer = [counts_layer, counts_layer]

        self.X_1: Union[np.ndarray, spmatrix] = adata_1.layers[counts_layer[0]] if counts_layer[0] else adata_1.X
        self.X_2: Union[np.ndarray, spmatrix] = adata_2.layers[counts_layer[1]] if counts_layer[1] else adata_2.X

        self.is_sparse_1: bool = isinstance(self.X_1, spmatrix)
        self.is_sparse_2: bool = isinstance(self.X_2, spmatrix)

        if shuffle:
            self.rng: Union[None, np.random.Generator] = rng or np.random.default_rng()
        else:
            self.rng: Union[None, np.random.Generator] = None
        self.shuffle: bool = shuffle

        self.library_size_1: Union[spmatrix, np.ndarray] = self.X_1.sum(1) if self.is_sparse_1 else self.X_1.sum(1, keepdims=True)
        self.library_size_2: Union[spmatrix, np.ndarray] = self.X_2.sum(1) if self.is_sparse_2 else self.X_2.sum(1, keepdims=True)

        if transformed_obsm is not None:
            if isinstance(transformed_obsm, str):
                transformed_obsm = [transformed_obsm, transformed_obsm]
            self.X_1_transformed = adata_1.obsm[transformed_obsm[0]] if transformed_obsm[0] else adata_1.X
            self.X_2_transformed = adata_2.obsm[transformed_obsm[1]] if transformed_obsm[1] else adata_2.X
        else:
            self.X_1_transformed = adata_1.X
            self.X_2_transformed = adata_2.X

        self.sample_batch_id: bool = sample_batch_id
        if self.sample_batch_id:
            assert batch_col in adata_1.obs, f'{batch_col} not in adata.obs'
            self.batch_indices: pd.Series = adata_1.obs[batch_col].astype('category').cat.codes
            
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

        if not self.require_counts:
            X_1 = torch.zeros([])
        elif isinstance(self.X_1, spmatrix):
            X_1 = torch.FloatTensor(self.X_1.todense())
        else:
            X_1 = torch.FloatTensor(self.X_1)
        
        if not self.require_counts:
            X_2 = torch.zeros([])
        elif isinstance(self.X_2, spmatrix):
            X_2 = torch.FloatTensor(self.X_2.todense())
        else:
            X_2 = torch.FloatTensor(self.X_2)
        library_size_1 = torch.FloatTensor(self.library_size_1)
        library_size_2 = torch.FloatTensor(self.library_size_2)
        if isinstance(self.X_1_transformed, spmatrix):
            X_1_transformed = torch.FloatTensor(self.X_1_transformed.todense())
        else:
            X_1_transformed = torch.FloatTensor(self.X_1_transformed)
        if isinstance(self.X_2_transformed, spmatrix):
            X_2_transformed = torch.FloatTensor(self.X_2_transformed.todense())
        else:
            X_2_transformed = torch.FloatTensor(self.X_2_transformed)
        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)
        result_dict = dict(cells_1=X_1, cells_2=X_2,
                        library_size_1=library_size_1,
                        library_size_2=library_size_2,
                        cells_1_transformed=X_1_transformed,
                        cells_2_transformed=X_2_transformed,
                        cell_indices=cell_indices)
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

            library_size_1 = torch.FloatTensor(self.library_size_1[batch])
            library_size_2 = torch.FloatTensor(self.library_size_2[batch])

            if not self.require_counts:
                cells_1 = cells_2 = torch.zeros([])
            else:
                X_1 = self.X_1[batch]
                X_2 = self.X_2[batch]

                if isinstance(X_1, spmatrix):
                    cells_1 = torch.FloatTensor(X_1.todense())
                else:
                    cells_1 = torch.FloatTensor(X_1)
                if isinstance(X_2, spmatrix):
                    cells_2 = torch.FloatTensor(X_2.todense())
                else:
                    cells_2 = torch.FloatTensor(X_2)

            X_1_transformed = self.X_1_transformed[batch]
            X_2_transformed = self.X_2_transformed[batch]
            if isinstance(X_1_transformed, spmatrix):
                X_1_transformed = torch.FloatTensor(X_1_transformed.todense())
            else:
                X_1_transformed = torch.FloatTensor(X_1_transformed)
            if isinstance(X_2_transformed, spmatrix):
                X_2_transformed = torch.FloatTensor(X_2_transformed.todense())
            else:
                X_2_transformed = torch.FloatTensor(X_2_transformed)

            cell_indices = torch.LongTensor(batch)
            result_dict = dict(cells_1=cells_1, cells_2=cells_2,
                        library_size_1=library_size_1,
                        library_size_2=library_size_2,
                        cells_1_transformed=X_1_transformed,
                        cells_2_transformed=X_2_transformed,
                        cell_indices=cell_indices)
            if self.sample_batch_id:
                result_dict['batch_indices'] = torch.LongTensor(self.batch_indices[batch])
            yield result_dict


class TriCellSampler():
    """An iterable cell dataset for minibatch sampling.

    Attributes:
        n_cells: number of cells in the dataset.
        batch_size: size of each sampled minibatch.
        n_epochs: number of epochs to sample before raising StopIteration.
        X_1: a (dense or sparse) matrix containing the cell-first_modality matrix.
        X_2: a (dense or sparse) matrix containing the cell-second_modality matrix.
        X_1_transformed: X_1 after applying the first modality transformation.
        X_2_transformed: X_2 after applying the second modality transformation.
        is_sparse_1: whether self.X_1 is a sparse matrix.
        is_sparse_2: whether self.X_2 is a sparse matrix
        shuffle: whether to shuffle the dataset at the beginning of each epoch.
            When n_cells >= batch_size, this attribute is ignored.
        rng: the random number generator. Could be None if shuffle is False.
        library_size_1: a (dense or sparse) vector storing the first modality library size for
            each cell.
        library_size_2: a (dense or sparse) vector storing the second modality library
            size for each cell.
        sample_batch_id: whether to yield batch indices in each sample.
        batch_indices: a (dense or sparse) vector storing the batch indices
            for each cell. Only present if sample_batch_id is True.
    """

    def __init__(self,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        adata_3: anndata.AnnData,
        batch_size: int,
        counts_layer: List[Union[str, None]] = [None, None, None],
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        sample_batch_id: bool = False,
        require_counts: bool = True,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        batch_col: str = 'batch_indices',
        shuffle: bool = True
    ) -> None:
        """Initializes the CellSampler object.

        Args:
            adata_1: an AnnData object storing the dataset of the first modality.
            adata_2: an AnnData object storing the dataset of the second modality.
            batch_size: size of each sampled minibatch.
            counts_layer: AnnData layer corresponding to raw counts. If it is a singular str,
                the same counts_layer will be applied to both AnnDatas. If it is a list,
                the first will be applied to adata_1, second applied to adata_2
            transformed_obsm: AnnData obsm key corresponding to transformed data.
            sample_batch_id: whether to yield batch indices in each sample.
            require_counts: Whether the raw counts are required by the model. The raw counts are not needed
                if the model is not going to decode anything
            n_epochs: number of epochs to sample before raising StopIteration.
            rng: the random number generator.
                Could be None if shuffle is False.
            batch_col: a key in adata.obs to the batch column. Only used when
                sample_batch_id is True.
            shuffle: whether to shuffle the dataset at the beginning of each
                epoch. When n_cells >= batch_size, this attribute is ignored.
        
        Assumption that adata_1 and adata_2 have the same cells in the same order.
        """
        assert adata_1.n_obs == adata_2.n_obs == adata_3.n_obs, "The two AnnData objects have a different number of observations"
        self.n_cells: int = adata_1.n_obs
        self.batch_size: int = batch_size
        self.n_epochs: Union[int, float] = n_epochs
        self.require_counts: bool = require_counts

        if isinstance(counts_layer, str) or counts_layer is None:
            counts_layer = [counts_layer, counts_layer, counts_layer]

        self.X_1: Union[np.ndarray, spmatrix] = adata_1.layers[counts_layer[0]] if counts_layer[0] else adata_1.X
        self.X_2: Union[np.ndarray, spmatrix] = adata_2.layers[counts_layer[1]] if counts_layer[1] else adata_2.X
        self.X_3: Union[np.ndarray, spmatrix] = adata_3.layers[counts_layer[2]] if counts_layer[2] else adata_3.X

        self.is_sparse_1: bool = isinstance(self.X_1, spmatrix)
        self.is_sparse_2: bool = isinstance(self.X_2, spmatrix)
        self.is_sparse_3: bool = isinstance(self.X_3, spmatrix)

        if shuffle:
            self.rng: Union[None, np.random.Generator] = rng or np.random.default_rng()
        else:
            self.rng: Union[None, np.random.Generator] = None
        self.shuffle: bool = shuffle
        
        self.library_size_1: Union[spmatrix, np.ndarray] = self.X_1.sum(1) if self.is_sparse_1 else self.X_1.sum(1, keepdims=True)
        self.library_size_2: Union[spmatrix, np.ndarray] = self.X_2.sum(1) if self.is_sparse_2 else self.X_2.sum(1, keepdims=True)
        self.library_size_3: Union[spmatrix, np.ndarray] = self.X_3.sum(1) if self.is_sparse_3 else self.X_3.sum(1, keepdims=True)

        if transformed_obsm is not None:
            if isinstance(transformed_obsm, str):
                transformed_obsm = [transformed_obsm, transformed_obsm, transformed_obsm]
            self.X_1_transformed = adata_1.obsm[transformed_obsm[0]] if transformed_obsm[0] else adata_1.X
            self.X_2_transformed = adata_2.obsm[transformed_obsm[1]] if transformed_obsm[1] else adata_2.X
            self.X_3_transformed = adata_3.obsm[transformed_obsm[2]] if transformed_obsm[2] else adata_3.X
        else:
            self.X_1_transformed = adata_1.X
            self.X_2_transformed = adata_2.X
            self.X_3_transformed = adata_3.X

        self.sample_batch_id: bool = sample_batch_id
        if self.sample_batch_id:
            assert batch_col in adata_1.obs, f'{batch_col} not in adata.obs'
            self.batch_indices: pd.Series = adata_1.obs[batch_col].astype('category').cat.codes
            
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

        if not self.require_counts:
            X_1 = torch.zeros([])
            X_2 = torch.zeros([])
        else:
            X_1 = torch.FloatTensor(self.X_1.todense()) if isinstance(self.X_1, spmatrix) else torch.FloatTensor(self.X_1)
            X_2 = torch.FloatTensor(self.X_2.todense()) if isinstance(self.X_2, spmatrix) else torch.FloatTensor(self.X_2)
            X_3 = torch.FloatTensor(self.X_3.todense()) if isinstance(self.X_3, spmatrix) else torch.FloatTensor(self.X_3)

        library_size_1 = torch.FloatTensor(self.library_size_1)
        library_size_2 = torch.FloatTensor(self.library_size_2)
        library_size_3 = torch.FloatTensor(self.library_size_3)

        X_1_transformed = torch.FloatTensor(self.X_1_transformed.todense() if isinstance(self.X_1_transformed, spmatrix) else self.X_1_transformed)
        X_2_transformed = torch.FloatTensor(self.X_2_transformed.todense() if isinstance(self.X_2_transformed, spmatrix) else self.X_2_transformed)
        X_3_transformed = torch.FloatTensor(self.X_3_transformed.todense() if isinstance(self.X_3_transformed, spmatrix) else self.X_3_transformed)

        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)
        result_dict = dict(
            cells_1=X_1, cells_2=X_2, cells_3=X_3,
            library_size_1=library_size_1, library_size_2=library_size_2, library_size_3=library_size_3,
            cells_1_transformed=X_1_transformed, cells_2_transformed=X_2_transformed, cells_3_transformed=X_3_transformed,
            cell_indices=cell_indices
        )
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

            library_size_1 = torch.FloatTensor(self.library_size_1[batch])
            library_size_2 = torch.FloatTensor(self.library_size_2[batch])
            library_size_3 = torch.FloatTensor(self.library_size_3[batch])

            if not self.require_counts:
                cells_1 = cells_2 = cells_3 = torch.zeros([])
            else:
                X_1 = self.X_1[batch, :]
                X_2 = self.X_2[batch, :]
                X_3 = self.X_3[batch, :]

                cells_1 = torch.FloatTensor(X_1.todense() if isinstance(X_1, spmatrix) else X_1)
                cells_2 = torch.FloatTensor(X_2.todense() if isinstance(X_2, spmatrix) else X_2)
                cells_3 = torch.FloatTensor(X_3.todense() if isinstance(X_3, spmatrix) else X_3)

            X_1_transformed = self.X_1_transformed[batch, :]
            X_2_transformed = self.X_2_transformed[batch, :]
            X_3_transformed = self.X_3_transformed[batch, :]

            X_1_transformed = torch.FloatTensor(X_1_transformed.todense() if isinstance(X_1_transformed, spmatrix) else X_1_transformed)
            X_2_transformed = torch.FloatTensor(X_2_transformed.todense() if isinstance(X_2_transformed, spmatrix) else X_2_transformed)
            X_3_transformed = torch.FloatTensor(X_3_transformed.todense() if isinstance(X_3_transformed, spmatrix) else X_3_transformed)

            cell_indices = torch.LongTensor(batch)
            result_dict = dict(
                cells_1=cells_1, cells_2=cells_2, cells_3=cells_3,
                library_size_1=library_size_1, library_size_2=library_size_2, library_size_3=library_size_3,
                cells_1_transformed=X_1_transformed, cells_2_transformed=X_2_transformed, cells_3_transformed=X_3_transformed,
                cell_indices=cell_indices
            )
            if self.sample_batch_id:
                result_dict['batch_indices'] = torch.LongTensor(self.batch_indices[batch])
            yield result_dict
