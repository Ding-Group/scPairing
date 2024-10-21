import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from anndata import AnnData

from .batch_sampler import CellSampler
from .models import constants as constants
from .models.trimodel import Modalities, ModalityNumber, Trimodel
from .models.utils import set_seed
from .trainers.UnsupervisedTrainer import UnsupervisedTrainer

_logger = logging.getLogger(__name__)

template_str = \
"""scPairing model
mod1: {}
    low-dimensional representations: {}
    counts layer: {}
mod2: {}
    low-dimensional representations: {}
    counts layer: {}
mod3: {}
    low-dimensional representations: {}
    counts layer: {}
"""


class triscPairing:
    """Multimodal data integration using contrastive learning and variational inference.

    Parameters
    ----------
    adata1
        AnnData object corresponding to the first modality of a multimodal single-cell dataset
    adata2
        AnnData object corresponding to the second modality of a multimodal single-cell dataset
    adata3
        AnnData object corresponding to the third modality of a multimodal single-cell dataset
    mod1_type
        The modality type of adata1. One of the following:

        * ``'rna'`` - for scRNA-seq data modeled with a negative binomial distribution
        * ``'atac'`` - for scATAC-seq data modeled with a Bernoulli distribution
        * ``'protein'`` for epitope data modeled with a negative binomial distribution
        * ``'other'`` for other data modalities modeled with a Gaussian distribution
    mod2_type
        The modality type of adata2. The options are identical to mod1_type.
    mod3_type
        The modality type of adata3. The options are identical to mod1_type.
    batch_col
        Column in ``adata1.obs``, ``adata2.obs``, and ``adata3.obs`` corresponding to batch information.
    transformed_obsm
        Key(s) in ``adata1.obsm``, ``adata2.obsm``, and ``adata3.obsm`` corresponding to the low-dimension
        representations of each individual modality. If a string is provided, the same key will
        be applied to all three AnnData ``.obsm``s.
    counts_layer
        Key(s) in ``adata1.layers``, ``adata2.layers``, and ``adata3.layers`` corresponding to the raw counts for each modality.
        If a string is provided, the same key will be applied to both all three AnnData `.layers`.
        If ``None`` is provided, raw counts will be taken from the `X` slot.
    use_decoder
        Whether to train a decoder to reconstruct the counts on top of the low-dimension representations.
    emb_dim
        Dimension of the hyperspherical latent space
    encoder_hidden_dims
        Number of nodes and depth of the encoder
    decoder_hidden_dims
        Number of nodes and depth of the decoder
    reconstruct_mod1_fn
        Custom function that reconstructs the counts from the reconstructed low-dimension representations
        for the first data modality.
    reconstruct_mod2_fn
        Custom function that reconstructs the counts from the reconstructed low-dimension representations
        for the second modality.
    reconstruct_mod2_fn
        Custom function that reconstructs the counts from the reconstructed low-dimension representations
        for the third modality.
    seed
        Random seed for model reproducibility.
    **model_kwargs
        Keyword args for triscPairing.
    """
    trainer = None

    def __init__(
        self,
        adata1: AnnData,
        adata2: AnnData,
        adata3: AnnData,
        mod1_type: Modalities = 'rna',
        mod2_type: Modalities = 'atac',
        mod3_type: Modalities = 'protein',
        batch_col: Optional[str] = None,
        transformed_obsm: Union[str, List[str]] = 'X_pca',
        counts_layer: Optional[Union[str, List[str]]] = None,
        use_decoder: bool = False,
        emb_dim: int = 10,
        encoder_hidden_dims: Sequence[int] = (128,),
        decoder_hidden_dims: Sequence[int] = (128,),
        reconstruct_mod1_fn: Optional[Callable] = None,
        reconstruct_mod2_fn: Optional[Callable] = None,
        reconstruct_mod3_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        **model_kwargs
    ) -> None:
        if adata1.n_obs != adata2.n_obs or adata1.n_obs != adata3.n_obs:
            raise ValueError("The AnnData objects have different numbers of cells.")

        self.seed: Optional[int] = seed
        if seed:
            set_seed(seed)

        if isinstance(transformed_obsm, str):
            transformed_obsm = [transformed_obsm, transformed_obsm, transformed_obsm]
        if isinstance(counts_layer, str) or counts_layer is None:
            counts_layer = [counts_layer, counts_layer, counts_layer]
        self.transformed_obsm: List[str] = transformed_obsm
        self.counts_layer: List[Optional[str]] = counts_layer

        mod1_input_dim = adata1.obsm[transformed_obsm[0]].shape[1] if transformed_obsm[0] is not None else adata1.X.shape[1]
        mod2_input_dim = adata2.obsm[transformed_obsm[1]].shape[1] if transformed_obsm[1] is not None else adata2.X.shape[1]
        mod3_input_dim = adata3.obsm[transformed_obsm[2]].shape[1] if transformed_obsm[2] is not None else adata3.X.shape[1]

        if batch_col is not None and (batch_col not in adata1.obs_keys() or batch_col not in adata2.obs_keys()):
            raise ValueError("batch_col was not found in adata1.obs or adata2.obs")
        self.batch_col = batch_col
        n_batches = len(adata1.obs[batch_col].cat.categories) if batch_col is not None else 1
        if batch_col is not None and len(adata1.obs[batch_col].cat.categories) != len(adata2.obs[batch_col].cat.categories):
            raise ValueError("The number of batches present in adata1 and adata2 are different.")

        self.adata1: AnnData = adata1
        self.adata2: AnnData = adata2
        self.adata3: AnnData = adata3
        self.mod1_type: Modalities = mod1_type
        self.mod2_type: Modalities = mod2_type
        self.mod3_type: Modalities = mod3_type
        self.model: Trimodel = Trimodel(
            mod1_input_dim,
            mod2_input_dim,
            mod3_input_dim,
            adata1.n_vars,
            adata2.n_vars,
            adata3.n_vars,
            n_batches=n_batches,
            mod1_type=mod1_type,
            mod2_type=mod2_type,
            mod3_type=mod3_type,
            use_decoder=use_decoder,
            emb_dim=emb_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            reconstruct_mod1_fn=reconstruct_mod1_fn,
            reconstruct_mod2_fn=reconstruct_mod2_fn,
            reconstruct_mod3_fn=reconstruct_mod3_fn,
            **model_kwargs
        )

    def train(
        self,
        epochs: int,
        batch_size: int = 2000,
        restart_training: bool = False,
        **trainer_kwargs
    ) -> None:
        """Train the model.
        
        Parameters
        ----------
        epochs
            Number of epochs to train the model.
        batch_size
            Minibatch size. Larger batch sizes recommended in contrastive learning.
        restart_training
            Whether to re-initialize model parameters and train from scratch, or
            to continue training from the current parameters.
        **trainer_kwargs
            Keyword arguments for UnsupervisedTrainer
        """
        need_reconstruction: bool = self.model.use_decoder and \
            (self.model.reconstruct_mod2_fn is None or self.model.reconstruct_mod1_fn is None or self.model.reconstruct_mod3_fn is None)

        if self.trainer is None or restart_training:
            self.trainer: UnsupervisedTrainer = UnsupervisedTrainer(
                self.model,
                self.adata1,
                self.adata2,
                self.adata3,
                counts_layer=self.counts_layer,
                transformed_obsm=self.transformed_obsm,
                batch_size=batch_size,
                **trainer_kwargs
            )

        self.trainer.train(
            n_epochs=epochs,
            need_reconstruction=need_reconstruction,
            batch_col=self.batch_col,
            seed=self.seed
        )

    def get_latent_representation(
        self,
        adata1: Optional[AnnData] = None,
        adata2: Optional[AnnData] = None,
        adata3: Optional[AnnData] = None,
        batch_size: int = 2000,
    ) -> Tuple[np.array]:
        """Returns the embeddings for all three modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to first modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the second modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata3
            AnnData object corresponding to the third modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        batch_size
            Minibatch size.
        """
        self.model.eval()
        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        adata3 = self.adata3 if adata3 is None else adata3
        sampler = CellSampler(
            adata1, adata2, adata3,
            require_counts=self.model.use_decoder,
            counts_layer=self.counts_layer,
            transformed_obsm=self.transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.model.need_batch,
            n_epochs=1,
            batch_col=self.batch_col,
            shuffle=False
        )

        mod1_features = []
        mod2_features = []
        mod3_features = []
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())
            mod1_features.append(fwd_dict[constants.MOD1_EMB].detach().cpu())
            mod2_features.append(fwd_dict[constants.MOD2_EMB].detach().cpu())
            mod3_features.append(fwd_dict[constants.MOD3_EMB].detach().cpu())
        
        mod1_features = torch.cat(mod1_features, dim=0).numpy()
        mod2_features = torch.cat(mod2_features, dim=0).numpy()
        mod3_features = torch.cat(mod3_features, dim=0).numpy()
        return mod1_features, mod2_features, mod3_features

    def get_normalized_expression(
        self,
        adata1: Optional[AnnData] = None,
        adata2: Optional[AnnData] = None,
        adata3: Optional[AnnData] = None,
        batch_size: int = 2000
    ) -> Tuple[np.array]:
        """Returns the reconstructed counts for all three modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to the first modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the second modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata3
            AnnData object corresponding to the third modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        batch_size
            Minibatch size.
        """
        if not self.model.use_decoder and \
            (self.model.reconstruct_mod1_fn is None or self.model.reconstruct_mod2_fn is None or self.model.reconstruct_mod3_fn is None):
            raise ValueError(
                "Cannot compute reconstructed data when decoder is disabled " 
                "and custom reconstruction functions are not provided."
            )

        self.model.eval()

        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        adata3 = self.adata3 if adata3 is None else adata3
        sampler = CellSampler(
            adata1, adata2, adata3,
            require_counts=self.model.use_decoder,
            counts_layer=self.counts_layer,
            transformed_obsm=self.transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.model.need_batch,
            n_epochs=1,
            batch_col=self.batch_col,
            shuffle=False
        )

        mod1_reconstruct = []
        mod2_reconstruct = []
        mod3_reconstruct = []
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())
            mod1_reconstruct.append(fwd_dict[constants.MOD1_RECONSTRUCT].detach().cpu())
            mod2_reconstruct.append(fwd_dict[constants.MOD2_RECONSTRUCT].detach().cpu())
            mod3_reconstruct.append(fwd_dict[constants.MOD3_RECONSTRUCT].detach().cpu())
        
        mod1_reconstruct = torch.cat(mod1_reconstruct, dim=0).numpy()
        mod2_reconstruct = torch.cat(mod2_reconstruct, dim=0).numpy()
        mod3_reconstruct = torch.cat(mod3_reconstruct, dim=0).numpy()
        return mod1_reconstruct, mod2_reconstruct, mod3_reconstruct
    
    def get_likelihoods(
        self,
        adata1: Optional[AnnData] = None,
        adata2: Optional[AnnData] = None,
        adata3: Optional[AnnData] = None,
        batch_size: int = 2000
    ) -> Tuple[np.array]:
        """Return the likelihoods for all three modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to the first modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the second modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata3
            AnnData object corresponding to the third modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        batch_size
            Minibatch size.
        """
        if not self.model.use_decoder:
            _logger.warning(
                "The model has the full decoder disabled. "
                "The full reconstruction losses will be all zero."
            )
        if self.model.use_decoder and \
            (self.model.reconstruct_mod1_fn is not None or self.model.reconstruct_mod2_fn is not None or self.model.reconstruct_mod3_fn is not None):
            _logger.warning(
                "The model has custom reconstruction functions. "
                "The full reconstruction losses will be all zero."
            )
        self.model.eval()

        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        adata3 = self.adata3 if adata3 is None else adata3
        sampler = CellSampler(
            adata1, adata2, adata3,
            require_counts=self.model.use_decoder,
            counts_layer=self.counts_layer,
            transformed_obsm=self.transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.model.need_batch,
            n_epochs=1,
            batch_col=self.batch_col,
            shuffle=False
        )

        mod1_nll = torch.zeros([])
        mod2_nll = torch.zeros([])
        mod3_nll = torch.zeros([])
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())

            n_cells = data_dict['cells_1_transformed'].shape[0]

            mod1_nll += fwd_dict[constants.MOD1_LOSS] * n_cells
            mod2_nll += fwd_dict[constants.MOD2_LOSS] * n_cells
            mod3_nll += fwd_dict[constants.MOD3_LOSS] * n_cells
            mod1_nll += fwd_dict[constants.MOD1_RECONSTRUCTION_LOSS] * n_cells
            mod2_nll += fwd_dict[constants.MOD2_RECONSTRUCTION_LOSS] * n_cells
            mod3_nll += fwd_dict[constants.MOD3_RECONSTRUCTION_LOSS] * n_cells
        return {
            constants.MOD1_LOSS: mod1_nll.detach().numpy() / adata1.n_obs,
            constants.MOD2_LOSS: mod2_nll.detach().numpy() / adata2.n_obs,
            constants.MOD3_LOSS: mod3_nll.detach().numpy() / adata3.n_obs
        }

    def get_cross_modality_expression(
        self,
        from_modality: ModalityNumber,
        to_modality: ModalityNumber,
        adata: Optional[AnnData] = None,
        batch_size: int = 2000,
    ) -> np.array:
        """
        Predict the expression of the another modality given the representation
        of one modality.

        Parameters
        ----------
        from_modality
            Which modality to use as prediction input. One of:
            
            * ``'mod1'`` for the first modality
            * ``'mod2'`` for the second modality
            * ``'mod3'`` for the third modality
        to_modality
            Which modality whose expression will be predicted. Options are the
            same as ``from_modality``.
        adata
            AnnData object corresponding to one of the modalities in a multimodal
            dataset. If adata is None, use the AnnData object corresponding to the
            ``from_modality``.
        batch_size
            Minibatch size.
        """
        if not self.model.use_decoder:
            if to_modality == 'mod2' and self.model.reconstruct_mod2_fn is None:
                _logger.warning(
                    "The model has no decoder and no custom reconstruction function "
                    f"for the second modality. The reconstructed {self.transformed_obsm[1]} "
                    "will be returned."
                )
            elif to_modality == 'mod1' and self.model.reconstruct_mod1_fn is None:
                _logger.warning(
                    "The model has no decoder and no custom reconstruction function "
                    f"for the first modality. The reconstructed {self.transformed_obsm[0]} "
                    "will be returned."
                )
            elif to_modality == 'mod3' and self.model.reconstruct_mod3_fn is None:
                _logger.warning(
                    "The model has no decoder and no custom reconstruction function "
                    f"for the third modality. The reconstructed {self.transformed_obsm[2]} "
                    "will be returned."
                )

        if from_modality == 'mod1':
            if adata is None:
                adata = self.adata1
            transformed_obsm = self.transformed_obsm[0]
        elif from_modality == 'mod2':
            if adata is None:
                adata = self.adata2
            transformed_obsm = self.transformed_obsm[1]
        elif from_modality == 'mod3':
            if adata is None:
                adata = self.adata3
            transformed_obsm = self.transformed_obsm[2]

        self.model.eval()
        sampler = CellSampler(
            adata,
            transformed_obsm=transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.model.need_batch,
            n_epochs=1,
            batch_col=self.batch_col,
            shuffle=False
        )
        
        reconstructs = []
        latents = []
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model.intermodality_prediction(data_dict, from_modality, to_modality)
            reconstructs.append(fwd_dict['reconstruction'].detach().cpu())
            latents.append(fwd_dict['latents'].detach().cpu())
        reconstructs = torch.cat(reconstructs, dim=0).numpy()
        latents = torch.cat(latents, dim=0).numpy()
        return reconstructs, latents

    def save(
        self,
        dir_path: str,
        prefix: str = "",
        save_optimizer: bool = False,
        overwrite: bool = False
    ) -> None:
        """Save the model.

        Parameters
        ----------
        dir_path
            Directory to save the model to.
        prefix
            Prefix to prepend to file names.
        save_optimizer
            Whether to save the optimizer.
        overwrite
            Whether to overwrite existing directory.
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)

        torch.save(self.model.state_dict(), os.path.join(dir_path, f"{prefix}model.pt"))
        if save_optimizer:
            torch.save(self.trainer.optimizer.state_dict(), os.path.join(dir_path, f"{prefix}opt.pt"))

    def load(
        self,
        dir_path: str,
        prefix: str = ""
    ) -> None:
        """Load an existing model

        Parameters
        ----------
        dir_path
            Directory to load the model from.
        prefix
            Prefix to prepend to file names.
        """
        path = os.path.join(dir_path, f"{prefix}model.pt")
        self.model.load_state_dict(torch.load(path))

    def __repr__(self) -> str:
        """Returns a string representation of the model."""
        return template_str.format(
            self.mod1_type,
            self.transformed_obsm[0] or 'X',
            self.counts_layer[0] or 'X',
            self.mod2_type,
            self.transformed_obsm[1] or 'X',
            self.counts_layer[1] or 'X',
            self.mod3_type,
            self.transformed_obsm[2] or 'X',
            self.counts_layer[2] or 'X'
        )
