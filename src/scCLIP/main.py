from typing import Callable, List, Optional, Sequence, Tuple, Union
import logging
import os

from anndata import AnnData
import numpy as np
import torch
import random

from models.model import Model, Modalities
from trainers.UnsupervisedTrainer import UnsupervisedTrainer
from batch_sampler import CellSampler


_logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets the random seed to seed.
    Borrowed from https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031

    Parameters
    ----------
    seed
        The random seed.
    """
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class scPairing:
    """Multimodal data integration using contrastive learning and variational inference.

    Parameters
    ----------
    adata1
        AnnData object corresponding to the first modality of a multimodal single-cell dataset
    adata2
        AnnData object corresponding to the second modality of a multimodal single-cell dataset
    mod1_type
        The modality type of adata1. One of the following:

        * ``'rna'`` - for scRNA-seq data modeled with a negative binomial distribution
        * ``'atac'`` - for scATAC-seq data modeled with a Bernoulli distribution
        * ``'protein'`` for epitope data modeled with a negative binomial distribution
        * ``'other'`` for other data modalities modeled with a Gaussian distribution
    mod2_type
        The modality type of adata2. The options are identical to mod1_type.
    batch_col
        Column in ``adata1.obs`` and ``adata2.obs`` corresponding to batch information
    transformed_obsm
        Key(s) in ``adata1.obsm`` and ``adata2.obsm`` corresponding to the low-dimension
        representations of each individual modality. If a string is provided, the same key will
        be applied to both ``adata1.obsm`` and ``adata2.obsm``. If ``None`` is provided,
        the representations will be taken from ``adata1.X`` and/or ``adata2.X``.
    counts_layer
        Key(s) in ``adata1.layers`` and ``adata2.layers`` corresponding to the raw counts for each modality.
        If a string is provided, the same key will be applied to both ``adata1.layers`` and ``adata2.layers``.
        If ``None`` is provided, raw counts will be taken from ``adata1.X`` and/or ``adata2.X``.
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
    seed
        Random seed for model reproducibility.
    **model_kwargs
        Keyword args for scCLIP.
    """
    trainer = None

    def __init__(
        self,
        adata1: AnnData,
        adata2: AnnData,
        mod1_type: Modalities = 'rna',
        mod2_type: Modalities = 'atac',
        batch_col: Optional[str] = None,
        transformed_obsm: Union[str, List[str]] = 'X_pca',
        counts_layer: Optional[Union[str, List[str]]] = None,
        use_decoder: bool = False,
        emb_dim: int = 10,
        encoder_hidden_dims: Sequence[int] = (128,),
        decoder_hidden_dims: Sequence[int] = (128,),
        reconstruct_mod1_fn: Optional[Callable] = None,
        reconstruct_mod2_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
        **model_kwargs
    ) -> None:
        if adata1.n_obs != adata2.n_obs:
            raise ValueError("The two AnnData objects have different numbers of cells.")

        self.seed = seed
        if seed:
            set_seed(seed)

        if isinstance(transformed_obsm, str):
            transformed_obsm = [transformed_obsm, transformed_obsm]
        if isinstance(counts_layer, str) or counts_layer is None:
            counts_layer = [counts_layer, counts_layer]
        self.transformed_obsm = transformed_obsm
        self.counts_layer = counts_layer

        mod1_input_dim = adata1.obsm[transformed_obsm[0]].shape[1] if transformed_obsm[0] is not None else adata1.X.shape[1]
        mod2_input_dim = adata2.obsm[transformed_obsm[1]].shape[1] if transformed_obsm[1] is not None else adata2.X.shape[1]

        if batch_col is not None and (batch_col not in adata1.obs_keys() or batch_col not in adata2.obs_keys()):
            raise ValueError("batch_col was not found in adata1.obs or adata2.obs")
        self.batch_col = batch_col
        n_batches = len(adata1.obs[batch_col].cat.categories) if batch_col is not None else 1
        if batch_col is not None and len(adata1.obs[batch_col].cat.categories) != len(adata2.obs[batch_col].cat.categories):
            raise ValueError("The number of batches present in adata1 and adata2 are different.")

        self.adata1 = adata1
        self.adata2 = adata2
        self.model = Model(
            mod1_input_dim,
            mod2_input_dim,
            adata1.n_vars,
            adata2.n_vars,
            n_batches=n_batches,
            mod1_type=mod1_type,
            mod2_type=mod2_type,
            use_decoder=use_decoder,
            emb_dim=emb_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            reconstruct_mod1_fn=reconstruct_mod1_fn,
            reconstruct_mod2_fn=reconstruct_mod2_fn,
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
            (self.model.reconstruct_mod2_fn is None or self.model.reconstruct_mod1_fn is None)

        if self.trainer is None or restart_training:
            self.trainer: UnsupervisedTrainer = UnsupervisedTrainer(
                self.model,
                self.adata1,
                self.adata2,
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
        adata2: Optional[AnnData]= None,
        batch_size: int = 2000,
    ) -> Tuple[np.array]:
        """Returns the embeddings for both modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to one modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the other modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        batch_size
            Minibatch size.
        """
        self.model.eval()
        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        sampler = CellSampler(
            adata1, adata2,
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
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())
            mod1_features.append(fwd_dict['mod1_features'].detach().cpu())
            mod2_features.append(fwd_dict['mod2_features'].detach().cpu())
        
        mod1_features = torch.cat(mod1_features, dim=0).numpy()
        mod2_features = torch.cat(mod2_features, dim=0).numpy()
        return mod1_features, mod2_features

    def get_normalized_expression(
        self,
        adata1: Optional[AnnData] = None,
        adata2: Optional[AnnData] = None,
        batch_size: int = 2000
    ) -> Tuple[np.array]:
        """Returns the reconstructed counts for both modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to one modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the other modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        batch_size
            Minibatch size.
        """
        if not self.model.use_decoder and (self.model.reconstruct_mod1_fn is None or self.model.reconstruct_mod2_fn is None):
            raise ValueError(
                "Cannot compute reconstructed data when decoder is disabled " 
                "and custom reconstruction functions are not provided."
            )

        self.model.eval()

        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        sampler = CellSampler(
            adata1, adata2,
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
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())
            mod1_reconstruct.append(fwd_dict['mod1_reconstruct'].detach().cpu())
            mod2_reconstruct.append(fwd_dict['mod2_reconstruct'].detach().cpu())
        
        mod1_reconstruct = torch.cat(mod1_reconstruct, dim=0).numpy()
        mod2_reconstruct = torch.cat(mod2_reconstruct, dim=0).numpy()
        return mod1_reconstruct, mod2_reconstruct
    
    def get_likelihoods(
        self,
        adata1: Optional[AnnData] = None,
        adata2: Optional[AnnData] = None,
        include_mse: bool = True,
        batch_size: int = 2000
    ) -> Tuple[np.array]:
        """Return the likelihoods for both modalities.

        Parameters
        ----------
        adata1
            AnnData object corresponding to one modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        adata2
            AnnData object corresponding to the other modality of a multimodal single-cell dataset.
            If not provided, the AnnData provided on initialization will be used.
        TODO: Does include_mse make sense?
        batch_size
            Minibatch size.
        """
        if not include_mse and not self.model.use_decoder:
            _logger.warning(
                "The model has the full decoder disabled. "
                "The full reconstruction losses will be all zero."
            )
        if not include_mse and (self.model.reconstruct_mod1_fn is not None or self.model.reconstruct_mod2_fn is not None):
            _logger.warning(
                "The model has custom reconstruction functions. "
                "The full reconstruction losses will be all zero."
            )
        self.model.eval()

        adata1 = self.adata1 if adata1 is None else adata1
        adata2 = self.adata2 if adata2 is None else adata2
        sampler = CellSampler(
            adata1, adata2,
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
        for data_dict in sampler:
            data_dict = {k: v.to(self.model.device) for k, v in data_dict.items()}
            fwd_dict = self.model(data_dict, hyper_param_dict=dict())
            if include_mse:
                mod1_nll += fwd_dict['nb']
                mod2_nll += fwd_dict['bernoulli']
            mod1_nll += fwd_dict['mod1_reconstruct_loss']
            mod2_nll += fwd_dict['mod2_reconstruct_loss']
        return mod1_nll.detach().numpy(), mod2_nll.detach().numpy()

    def get_cross_modality_expression(
        self,
        mod1_to_mod2: bool,
        adata: Optional[AnnData] = None,
        batch_size: int = 2000,
    ) -> np.array:
        """
        Predict the expression of the other modality given the representation
        of one modality.

        Parameters
        ----------
        mod1_to_mod2
            Whether to compute the cross modality prediction from the first modality
            to the second modality.
        adata
            AnnData object corresponding to one of the modalities in a multimodal
            dataset. If adata is None and ``mod1_to_mod2`` is ``True``, then ``adata1``
            will be used. If adata is None and ``mod1_to_mod2`` is ``False``, then
            ``adata2`` will be used.
        batch_size
            Minibatch size.
        """
        if not self.model.use_decoder:
            if mod1_to_mod2 and self.model.reconstruct_mod2_fn is None:
                _logger.warning(
                    "The model has no decoder and no custom reconstruction function "
                    f"for the second modality. The reconstructed {self.transformed_obsm[1]} "
                    "will be returned."
                )
            elif not mod1_to_mod2 and self.model.reconstruct_mod1_fn is None:
                _logger.warning(
                    "The model has no decoder and no custom reconstruction function "
                    f"for the first modality. The reconstructed {self.transformed_obsm[0]} "
                    "will be returned."
                )

        if mod1_to_mod2 and adata is None:
            adata = self.adata1
        elif not mod1_to_mod2 and adata is None:
            adata = self.adata2

        transformed_obsm = self.transformed_obsm[0] if mod1_to_mod2 else self.transformed_obsm[1]

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
            fwd_dict = self.model.pred_mod1_mod2_forward(data_dict) if mod1_to_mod2 \
                else self.model.pred_mod2_mod1_forward(data_dict)
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

    # def __str__(self) -> str:
    #     """
    #     """
    #     pass
