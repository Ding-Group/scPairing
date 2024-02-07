from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import math
import anndata
import numpy as np
import logging
import random
from scipy.sparse import spmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent

from batch_sampler import CellSampler
from .log_likelihood import log_nb_positive
from .distributions import PowerSpherical, _kl_powerspherical_uniform
from .losses import ClipLoss, BatchClipLoss, DebiasedClipLoss, SigmoidLoss

Loss = Literal['clip', 'debiased_clip', 'batch_clip', 'sigmoid']
Modalities = Literal['rna', 'atac', 'protein', 'other']
Combine = Literal['dropout', 'average']


_logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets the random seed to seed.

    Parameters
    ----------
    seed
        The random seed.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def get_fully_connected_layers(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    norm_type: Literal['layer', 'batch', 'none'] = 'layer',
    dropout_prob: float = 0
) -> nn.Sequential:
    """Construct fully connected layers.
    """
    layers = []
    for i, size in enumerate(hidden_dims):
        lin = nn.Linear(input_dim, size)
        nn.init.xavier_normal_(lin.weight)
        layers.append(lin)
        layers.append(nn.ELU())
        if norm_type == 'layer':
            layers.append(nn.LayerNorm(size))
        elif norm_type == 'batch':
            layers.append(nn.BatchNorm1d(size))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class scCLIP(nn.Module):
    """Variational autoencoder model

    Parameters
    ----------
    mod1_input_dim
        Input dimension of the low-dimensional representations for the first data modality.
    mod2_input_dim
        Input dimension of the low-dimensional representations for the second data modality.
    mod1_var_dim
        Dimension of the count features for the first data modality.
    mod2_var_dim
        Dimension of the count features for the second data modality.
    n_batches
        Number of batches present in the dataset
    mod1_type
        The modality type of adata1. One of the following:

        * ``'rna'`` - for scRNA-seq data modeled with a negative binomial distribution
        * ``'atac'`` - for scATAC-seq data modeled with a Bernoulli distribution
        * ``'protein'`` for epitope data modeled with a negative binomial distribution
        * ``'other'`` for other data modalities modeled with a Gaussian distribution
    mod2_type
        The modality type of adata2. The options are identical to mod1_type.
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
    loss_method
        The type of contrastive loss to use. One of the following:

        * ``'clip'`` - contrastive loss from `Learning Transferable Visual Models From Natural Language Supervision` (Radford `et al.`, 2021)
        * ``'debiased_clip'`` - debiased contrastive loss from `Debiased Contrastive Learning` (Chuang `et al.`, 2020)
        * ``'batch_clip'`` - contrastive loss except applied on only cells of the same batch
        * ``'sigmoid'`` - sigmoid loss from `Sigmoid Loss for Language Image Pre-Training` (Zhai `et al.`, 2023)
    variational
        Whether the model should be a VAE or AE.
    combine_method
        Method for merging representations from the two encoders. One of the following:

        * ``'dropout'`` - Random dropout of one representation and replace the zeros with the values
          from the second representation, from `Cross-modal autoencoder framework learns holistic representations
          of cardiovascular state` (Radhakrishnan `et al.`, 2023).
        * ``'average'`` - Average between the two representations
    batch_dispersion
        Whether dispersion for negative binomial likelihood models is separate for different batches
    use_norm
        Type of normalization to apply. One of:

        * ``'layer'`` - layer normalization
        * ``'batch'`` - batch normalization
        * ``'none'`` - no normalization
    dropout
        Dropout proportion
    
    seed
        Random seed for model reproducibility.
    """
    emb_names: Sequence[str] = ['mod1_features', 'mod2_features']

    def __init__(
        self,
        mod1_input_dim: int,
        mod2_input_dim: int,
        mod1_var_dim: int,
        mod2_var_dim: int,
        n_batches: int = 1,
        mod1_type: Modalities = 'rna',
        mod2_type: Modalities = 'atac',
        use_decoder: bool = True,
        emb_dim: int = 10,
        encoder_hidden_dims: Sequence[int] = (128,),
        decoder_hidden_dims: Sequence[int] = (128,),
        reconstruct_mod1_fn: Optional[Callable] = None,
        reconstruct_mod2_fn: Optional[Callable] = None,
        loss_method: Loss = 'clip',
        variational: bool = True,
        combine_method: Combine = "dropout",
        batch_dispersion: bool = False,
        use_norm: Literal['layer', 'batch', 'none'] = 'batch',
        dropout: float = 0.1,
        modality_discriminative: bool = True,
        batch_discriminative: bool = False,
        batch_discriminative_weight: float = 1,
        distance_loss: bool = True,
        downsample_clip: bool = False,
        downsample_clip_prob: float = 0.5,
        tau: float = 0.1,
        set_temperature: Optional[float] = None,
        cap_temperature: Optional[float] = None,
        per_cell_temperature: bool = False,
        seed=None,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super().__init__()

        # Validation and warnings
        # Validate modality types
        if mod1_type not in ['rna', 'atac', 'protein', 'other']:
            raise ValueError("mod1_type must be one of 'rna', 'atac', 'protein', or 'other'")
        if mod2_type not in ['rna', 'atac', 'protein', 'other']:
            raise ValueError("mod1_type must be one of 'rna', 'atac', 'protein', or 'other'")
        # Validate batch choices
        if batch_dispersion and n_batches == 1:
            _logger.warning("With one batch provided, per-batch dispersion will be disabled")
            batch_dispersion = False
        if batch_discriminative and n_batches == 1:
            _logger.warning("With one batch provided, batch adversarial loss will be disabled")
            batch_discriminative = False
        # Validate decoder choices
        if (reconstruct_mod1_fn is not None or reconstruct_mod2_fn is not None) and not use_decoder:
            _logger.warning("Reconstruction functions were provided but decoding was turned off.\n"
                            "The provided reconstruction functions will not be used.")

        if seed:
            set_seed(seed)

        # Model parameters
        self.device: torch.device = device
        self.n_batches: int = n_batches
        self.need_batch: bool = n_batches > 1

        # Encoder input dimensions
        self.mod1_input_dim: int = mod1_input_dim
        self.mod2_input_dim: int = mod2_input_dim
        # Decoder output dimensions
        self.n_mod1_output: int = mod1_var_dim if reconstruct_mod1_fn is None else mod1_input_dim
        self.n_mod2_output: int = mod2_var_dim if reconstruct_mod2_fn is None else mod2_input_dim

        self.emb_dim: int = emb_dim
        self.encoder_hidden_dims: Sequence[int] = encoder_hidden_dims
        self.decoder_hidden_dims: Sequence[int] = decoder_hidden_dims
        self.norm: Literal['layer', 'batch', 'none'] = use_norm
        self.dropout: float = dropout

        self.mod1_type: Modalities = mod1_type
        self.mod2_type: Modalities = mod2_type

        # Encoder networks
        self.mod1_encoder: nn.Module = get_fully_connected_layers(
            self.mod1_input_dim,
            self.emb_dim + 1,
            self.encoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod2_encoder: nn.Module = get_fully_connected_layers(
            self.mod2_input_dim,
            self.emb_dim + 1,
            self.encoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )

        self.per_cell_temperature: bool = per_cell_temperature
        self.cap_temperature: Optional[float] = cap_temperature
        if set_temperature is not None:
            self.logit_scale: torch.Tensor = torch.ones([]) * set_temperature 
        elif per_cell_temperature:
            self.logit_scale_nn: nn.Module = nn.Sequential(
                nn.Linear(self.mod1_input_dim + self.mod2_input_dim, 16),
                nn.LayerNorm(16),
                nn.ELU(),
                nn.Linear(16, 1),
                nn.ReLU()
            )
        else:
            self.logit_scale: nn.Parameter = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.modality_discriminative: bool = modality_discriminative
        if self.modality_discriminative:
            self.discriminator: nn.Module = nn.Sequential(
                nn.Linear(self.emb_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Directly predict binary
            )
            self.bce_loss: nn.Module = nn.BCELoss()
        self.batch_discriminative: bool = batch_discriminative
        self.batch_discriminative_weight: float = batch_discriminative_weight
        if self.batch_discriminative:
            self.batch_discriminator = nn.Sequential(
                nn.Linear(self.emb_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, self.n_batches)
            )
            self.ce_loss: nn.Module = nn.CrossEntropyLoss()

        self.distance_loss: bool = distance_loss
        if self.distance_loss:
            self.cosine_loss: nn.Module = nn.CosineSimilarity()

        self.variational: bool = variational
        if variational:
            self.emb_indices: torch.Tensor = torch.arange(self.emb_dim, dtype=torch.float) + 1

        self.loss_method: Loss = loss_method
        if loss_method == 'clip':
            self.clip_loss = ClipLoss(downsample_clip=downsample_clip, downsample_clip_prob=downsample_clip_prob)
        elif loss_method == 'debiased_clip':
            self.clip_loss = DebiasedClipLoss(tau, downsample_clip=downsample_clip, downsample_prob=downsample_clip_prob)
        elif loss_method == 'sigmoid':
            self.clip_loss = SigmoidLoss()
            self.bias = nn.Parameter(torch.ones([]) * 0)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        elif loss_method == 'batch_clip':
            self.clip_loss = BatchClipLoss(tau)

        self.use_decoder: bool = use_decoder
        self.combine_method: Combine = combine_method

        self.reconstruct_mod1_fn: Optional[Callable] = reconstruct_mod1_fn
        self.reconstruct_mod2_fn: Optional[Callable] = reconstruct_mod2_fn
        self.batch_dispersion: bool = batch_dispersion
        self._init_decoders()

        self.device = device
        self.to(device)

    def _init_decoders(self):
        """Initialize the decoders
        """
        self.mod1_decoder: nn.Module = get_fully_connected_layers(
            self.emb_dim,
            self.mod1_input_dim,
            (128, ),  # TODO: Make this a hyperparameter
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod2_decoder: nn.Module = get_fully_connected_layers(
            self.emb_dim,
            self.mod2_input_dim,
            (128, ),  # TODO: Make this a hyperparameter
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        if self.use_decoder and self.reconstruct_mod1_fn is None:
            mod1_decoder_input_dim = self.mod1_input_dim + self.n_batches if self.reconstruct_mod1_fn is None and self.n_batches > 1 else self.mod1_input_dim
            self.mod1_reconstructor: nn.Module = get_fully_connected_layers(
                mod1_decoder_input_dim,
                self.n_mod1_output,
                self.decoder_hidden_dims,
                norm_type=self.norm,
                dropout_prob=self.dropout
            )
        if self.use_decoder and self.reconstruct_mod2_fn is None:
            mod2_decoder_input_dim = self.mod2_input_dim + self.n_batches if self.reconstruct_mod2_fn is None and self.n_batches > 1 else self.mod2_input_dim
            self.mod2_reconstructor: nn.Module = get_fully_connected_layers(
                mod2_decoder_input_dim,
                self.n_mod2_output,
                self.decoder_hidden_dims,
                norm_type=self.norm,
                dropout_prob=self.dropout
            )

        if self.reconstruct_mod1_fn is None and (self.mod1_type == 'rna' or self.mod1_type == 'protein'):
            if self.batch_dispersion:
                self.mod1_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_batches, self.n_mod1_output))
            else:
                self.mod1_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_mod1_output))
        if self.reconstruct_mod2_fn is None and (self.mod2_type == 'rna' or self.mod2_type == 'protein'):
            if self.batch_dispersion:
                self.mod2_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_batches, self.n_mod2_output))
            else:
                self.mod2_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_mod2_output))

    def decode_mod1(
        self,
        features: torch.Tensor,
        mod1_embs: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        cell_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        is_imputation: bool = False,
    ) -> Mapping[str, Any]:
        """Decode first modality data from the combined features.

        Parameters
        ----------
        features
            Embeddings on the common hyperspherical latent space.
        mod1_embs
            Low-dimension representation of the first data modality.
        counts
            Raw counts of the first data modality.
        library_size
            Library size of the first data modality.
        cell_indices
            Indices corresponding to the cells in the minibatch.
        batch_indices
            Tensor of integer batch labels.
        is_imputation
            Whether the model is decoding cells with known low-dimension representations
            and raw counts, or is imputing unseen data.
        """
        approx_mod1_features = self.mod1_decoder(features)
        loss = torch.nn.MSELoss()(approx_mod1_features, mod1_embs) if not is_imputation else torch.zeros([])
        if not self.use_decoder:
            return {
                'mod1_reconstruct': torch.zeros([]),
                'mod1_reconstruct_loss': torch.zeros([]),
                'nll': loss
            }
        if self.reconstruct_mod1_fn is None:
            if self.reconstruct_mod1_fn is None and self.n_batches > 1:
                batch_one_hot = F.one_hot(batch_indices, num_classes=self.n_batches)
                if is_imputation:
                    batch_one_hot = torch.zeros(features.shape[0], self.n_batches)
                approx_mod1_features = torch.cat([approx_mod1_features, batch_one_hot], axis=1)
            approx_mod1_features = self.mod1_reconstructor(approx_mod1_features)
            if self.mod1_type == 'rna':
                if is_imputation:
                    library_size = torch.ones([])
                mod1_reconstruct = F.softmax(approx_mod1_features, dim=1) * library_size
                dispersion = self.mod1_dispersion[batch_indices] if self.batch_dispersion else self.mod1_dispersion
                reconstruction_loss = -log_nb_positive(
                    counts,
                    mod1_reconstruct,
                    dispersion.exp()
                ).mean() / self.n_mod1_output if not is_imputation else torch.zeros([])
            elif self.mod1_type == 'atac':
                mod1_reconstruct = torch.sigmoid(approx_mod1_features)
                reconstruction_loss = F.binary_cross_entropy(
                    mod1_reconstruct,
                    counts,
                    reduction="none"
                ).sum(-1).mean() * 10 / self.n_mod1_output if not is_imputation else torch.zeros([])
            elif self.mod1_type == 'other':
                mod1_reconstruct = approx_mod1_features
                reconstruction_loss = torch.nn.MSELoss()(counts, mod1_reconstruct).mean() / self.n_mod1_output if not is_imputation else torch.zeros([])
            loss += reconstruction_loss
        else:
            mod1_reconstruct, _ = self.reconstruct_mod1_fn(
                approx_mod1_features,
                mod1_embs,
                counts,
                library_size,
                cell_indices,
                self.training,
                is_imputation,
                batch_indices
            )
            reconstruction_loss = torch.zeros([])
        return {
            'mod1_reconstruct': mod1_reconstruct,
            'mod1_reconstruct_loss': reconstruction_loss,
            'nll': loss
        }

    def decode_mod2(
        self,
        features: torch.Tensor,
        mod2_embs: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        cell_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        is_imputation: bool = False,
    ) -> Mapping[str, Any]:
        """Decode second modality data from the combined features.

        Parameters
        ----------
        features
            Embeddings on the common hyperspherical latent space.
        mod2_embs
            Low-dimension representation of the second data modality.
        counts
            Raw counts of the second data modality.
        library_size
            Library size of the second data modality.
        cell_indices
            Indices corresponding to the cells in the minibatch.
        batch_indices
            Tensor of integer batch labels.
        is_imputation
            Whether the model is decoding cells with known low-dimension representations
            and raw counts, or is imputing unseen data.
        """
        approx_mod2_features = self.mod2_decoder(features)
        loss = torch.nn.MSELoss()(approx_mod2_features, mod2_embs) if not is_imputation else torch.zeros([])
        if not self.use_decoder:
            return {
                'mod2_reconstruct': torch.zeros([]),
                'mod2_reconstruct_loss': torch.zeros([]),
                'nll': loss
            }
        if self.reconstruct_mod2_fn is None:
            if self.reconstruct_mod2_fn is None and self.n_batches > 1:
                batch_one_hot = F.one_hot(batch_indices, num_classes=self.n_batches)
                if is_imputation:
                    batch_one_hot = torch.zeros(features.shape[0], self.n_batches)
                approx_mod2_features = torch.cat([approx_mod2_features, batch_one_hot], axis=1)
            approx_mod2_features = self.mod2_reconstructor(approx_mod2_features)
            if self.mod2_type == 'rna' or self.mod2_type == 'protein':
                if is_imputation:
                    library_size = torch.ones([])
                mod2_reconstruct = F.softmax(approx_mod2_features, dim=1) * library_size
                dispersion = self.mod2_dispersion[batch_indices] if self.batch_dispersion else self.mod2_dispersion
                reconstruction_loss = -log_nb_positive(
                    counts,
                    mod2_reconstruct,
                    dispersion.exp()
                ).mean() / self.n_mod2_output if not is_imputation else torch.zeros([])
            elif self.mod2_type == 'atac':
                mod2_reconstruct = torch.sigmoid(approx_mod2_features)
                reconstruction_loss = F.binary_cross_entropy(
                    mod2_reconstruct,
                    counts,
                    reduction="none"
                ).sum(-1).mean() * 10 / self.n_mod2_output if not is_imputation else torch.zeros([])
            elif self.mod2_type == 'other':
                mod2_reconstruct = approx_mod2_features
                reconstruction_loss = torch.nn.MSELoss()(counts, mod2_reconstruct).mean() / self.n_mod2_output if not is_imputation else torch.zeros([])
            loss += reconstruction_loss
        else:
            mod2_reconstruct, _ = self.reconstruct_mod2_fn(
                approx_mod2_features,
                mod2_embs,
                counts,
                library_size,
                cell_indices,
                self.training,
                is_imputation,
                batch_indices
            )
            reconstruction_loss = torch.zeros([])
        return {
            'mod2_reconstruct': mod2_reconstruct,
            'mod2_reconstruct_loss': reconstruction_loss,
            'nll': loss
        }

    def combine_features(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor
    ) -> torch.Tensor:
        """Combines the features using the ``combine_method`` of this model.

        Parameters
        ----------
        mod1_features
            Embeddings on the common hyperspherical latent space from the
            first modality encoded by the ``mod1_encoder``.
        mod2_features
            Embeddings on the common hyperspherical latent space from the
            second modality encoded by the ``mod2_encoder``.
        """
        if self.combine_method == 'average':
            return (mod1_features + mod2_features) / 2
        elif self.combine_method == 'dropout':
            mask = torch.rand(self.emb_indices.shape[0]) < 0.5
            mod1_features[:, mask] = 0
            mod2_features[:, ~mask] = 0
            combined = mod1_features + mod2_features
            return combined

    def forward(
        self,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any] = dict()
    ) -> Mapping[str, Any]:
        """Compute the forward pass.

        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        hyper_param_dict
            Dictionary containing hyperparameters.
        """
        counts_1, counts_2 = data_dict["cells_1"], data_dict["cells_2"]
        library_size_1, library_size_2 = data_dict["library_size_1"], data_dict["library_size_2"]
        cell_indices = data_dict['cell_indices']
        batch_indices = data_dict.get('batch_indices', None)

        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]

        mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)

        mod1_mu, mod1_var = mod1_features[:, :self.emb_dim], mod1_features[:, self.emb_dim:]
        mod2_mu, mod2_var = mod2_features[:, :self.emb_dim], mod2_features[:, self.emb_dim:]

        # L2 normalization
        mod1_features = F.normalize(mod1_mu)
        mod2_features = F.normalize(mod2_mu)

        combined_features = self.combine_features(mod1_features.clone(), mod2_features.clone())
        combined_features = combined_features / torch.norm(combined_features, p=2, dim=-1, keepdim=True)
        if self.variational:
            combined_var = nn.Softplus()((mod1_var + mod2_var) / 2) + 1e-5

            z_dist = PowerSpherical(combined_features, combined_var.squeeze(-1))
            if self.training:
                z = z_dist.rsample()
            else:
                z = combined_features
        else:
            z = combined_features

        if self.variational:
            mod1_dict = self.decode_mod1(z, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
            mod2_dict = self.decode_mod2(z, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
        else:
            mod1_dict = self.decode_mod1(mod2_features, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
            mod2_dict = self.decode_mod2(mod1_features, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
        log_px_zl = mod1_dict['nll'] + mod2_dict['nll']
        nb = mod1_dict['nll']
        bernoulli = mod2_dict['nll']


        if self.per_cell_temperature:
            logit_scale = self.logit_scale_nn(torch.cat([mod1_input, mod2_input], axis=1)).exp()
        else:
            logit_scale = self.logit_scale.exp()
        if self.cap_temperature is not None:
            logit_scale = torch.clamp(logit_scale, max=self.cap_temperature)

        fwd_dict = {
            "mod1_features": mod1_features,
            "mod2_features": mod2_features,
            "temp": logit_scale.mean(),
            "nll": log_px_zl.mean(),
            "mod1_reconstruct_loss": mod1_dict["mod1_reconstruct_loss"],
            "mod2_reconstruct_loss": mod2_dict["mod2_reconstruct_loss"],
            "nb": nb,
            "bernoulli": bernoulli
        }
        
        if self.variational and self.use_decoder:
            fwd_dict['combined_features'] = z
        if self.use_decoder:
            fwd_dict['mod1_reconstruct'] = mod1_dict['mod1_reconstruct']
            fwd_dict['mod2_reconstruct'] = mod2_dict['mod2_reconstruct']

        if not self.training:
            return fwd_dict

        if self.loss_method == 'sigmoid':
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale, self.bias)
            fwd_dict['bias'] = self.bias
        elif self.loss_method == 'batch_clip':
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale, batch_indices)
        else:
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale)
        fwd_dict['contrastive'] = contrastive_loss

        loss = log_px_zl + contrastive_loss

        if self.modality_discriminative and self.training:
            mod1_preds = self.discriminator(mod1_features)
            mod2_preds = self.discriminator(mod2_features)
            truth = torch.cat([torch.zeros(mod1_features.shape[0], device=self.device), torch.ones(mod2_features.shape[0], device=self.device)])
            discriminative_loss = self.bce_loss(torch.cat([mod1_preds, mod2_preds]).squeeze(-1), truth.squeeze(-1))
            fwd_dict['modality_discriminative'] = discriminative_loss

            loss = loss - discriminative_loss

        if self.batch_discriminative and self.training:
            batch_pred_1 = self.batch_discriminator(mod1_features)
            batch_pred_2 = self.batch_discriminator(mod2_features)
            truth = torch.cat([batch_indices, batch_indices])
            batch_loss = self.ce_loss(torch.cat([batch_pred_1, batch_pred_2]).squeeze(-1), truth.squeeze(-1))
            fwd_dict['batch_discriminative'] = batch_loss

            loss = loss - batch_loss * hyper_param_dict.get('batch_weight', 1)

        if self.distance_loss and self.training:
            dist_loss = self.cosine_loss(mod1_features, mod2_features).mean()
            # Want to encourage them to be similar
            loss = loss - dist_loss

            fwd_dict['dist'] = dist_loss

        if self.variational:
            uni = HypersphericalUniform(dim=self.emb_dim - 1, device=self.device)
            kl = _kl_powerspherical_uniform(z_dist, uni)
            # ps = PowerSpherical(mu, var.squeeze(-1))
            # kl = _kl_powerspherical_uniform(mod1_z_dist, uni) + _kl_powerspherical_uniform(mod2_z_dist, uni)
            fwd_dict['KL'] = kl.mean()
            loss += kl.mean() * hyper_param_dict.get('kl_weight', 1)
        else:
            fwd_dict['KL'] = torch.zeros([])

        record = dict(
            loss=loss,
            contrastive=fwd_dict['contrastive'],
            KL=fwd_dict['KL'],
            nb=fwd_dict['nb'],
            bernoulli=fwd_dict['bernoulli'],
            temp=fwd_dict['temp'],
        )
        if self.modality_discriminative and self.training:
            record['modality_discriminative'] = fwd_dict['modality_discriminative']
        if self.batch_discriminative and self.training:
            record['batch_discriminative'] = fwd_dict['batch_discriminative']
        if self.distance_loss and self.training:
            record['dist'] = fwd_dict['dist']
        if self.loss_method == 'sigmoid':
            record['bias'] = fwd_dict['bias']
        record = {k: v.detach().item() for k, v in record.items()}
        return loss, fwd_dict, record

    def discriminative_forward(self, data_dict):
        """Compute the discriminative loss

        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        """
        if not self.modality_discriminative and not self.batch_discriminative:
            return None
        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]
        batch_indices = data_dict.get('batch_indices', None)

        mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)
        mod1_mu, _ = mod1_features[:, :self.emb_dim], mod1_features[:, self.emb_dim:]
        mod2_mu, _ = mod2_features[:, :self.emb_dim], mod2_features[:, self.emb_dim:]

        # L2 normalization
        mod1_features = F.normalize(mod1_mu)
        mod2_features = F.normalize(mod2_mu)

        discriminative_loss = 0
        if self.modality_discriminative:
            mod1_preds = self.discriminator(mod1_features)
            mod2_preds = self.discriminator(mod2_features)
            truth = torch.cat([torch.zeros(mod1_features.shape[0], device=self.device), torch.ones(mod2_features.shape[0], device=self.device)])
            discriminative_loss = discriminative_loss + self.bce_loss(torch.cat([mod1_preds, mod2_preds]).squeeze(-1), truth.squeeze(-1))

        batch_loss = 0
        if self.batch_discriminative:
            batch_pred_1 = self.batch_discriminator(mod1_features)
            batch_pred_2 = self.batch_discriminator(mod2_features)
            truth = torch.cat([batch_indices, batch_indices])
            batch_loss = self.ce_loss(torch.cat([batch_pred_1, batch_pred_2]).squeeze(-1), truth.squeeze(-1))

            discriminative_loss = discriminative_loss + batch_loss
        return discriminative_loss

    def train_step(self,
        optimizers: List[optim.Optimizer],
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any],
        loss_update_callback: Union[None, Callable] = None
    ) -> Mapping[str, torch.Tensor]:
        """Executes a training step given a minibatch of data.

        Set the model to train mode, run the forward pass, back propagate the
        gradients, step the optimizer, return the record for this step.

        Parameters
        ----------
        optimizer
            Optimizer of the model parameters.
        data_dict
            Dictionary containing the minibatch training data.
        hyper_param_dict
            Dictionary containing hyperparameters.
        loss_update_callback
            Callable that updates the loss and the record dict.
        """
        self.train()
        optimizers[0].zero_grad()
        loss, fwd_dict, new_record = self(data_dict, hyper_param_dict)
        if loss_update_callback is not None:
            loss, new_record = loss_update_callback(loss, fwd_dict, new_record)
        loss.backward()
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        new_record['max_norm'] = norms.cpu().numpy()
        optimizers[0].step()

        if optimizers[1] is not None:
            optimizers[1].zero_grad()
            loss = self.discriminative_forward(data_dict)
            loss.backward()
            optimizers[1].step()
        return new_record

    def get_cell_embeddings_and_nll(
        self,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        counts_layer=None,
        transformed_obsm=None,
        inplace: bool = True
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """Calculates cell embeddings and negative log-likelihood

        Parameters
        ----------
        """
        nlls = []
        if not self.use_decoder:
            nlls = None
        if emb_names is None:
            emb_names = self.emb_names
        self.eval()
        if isinstance(emb_names, str):
            emb_names = [emb_names]

        embs = {name: [] for name in emb_names}
        hyper_param_dict = dict(decode=nlls is not None)

        def store_emb_and_nll(data_dict, fwd_dict):
            for name in emb_names:
                embs[name].append(fwd_dict[name].detach().cpu())
            if nlls is not None:
                nlls.append(fwd_dict['nll'].detach().item())

        self._apply_to(
            adata1, adata2, batch_col, batch_size,
            counts_layer=counts_layer,
            transformed_obsm=transformed_obsm,
            hyper_param_dict=hyper_param_dict,
            callback=store_emb_and_nll
        )

        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}
        if nlls is not None:
            nll = sum(nlls) / adata1.n_obs
        else:
            nll = None

        if inplace:
            adata1.obsm.update(embs)
            adata2.obsm.update(embs)
            return nll
        else:
            return embs, nll

    def _apply_to(self,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        batch_col: str = 'batch_indices',
        batch_size: int = 2000,
        counts_layer=None,
        transformed_obsm=None,
        hyper_param_dict: Union[dict, None] = None,
        callback: Union[Callable, None] = None
    ) -> None:
        """Docstring (TODO)
        """
        sampler = CellSampler(
            adata1, adata2,
            require_counts=self.use_decoder,
            counts_layer=counts_layer,
            transformed_obsm=transformed_obsm,
            batch_size=batch_size, sample_batch_id=self.need_batch,
            n_epochs=1, batch_col=batch_col, shuffle=False
        )
        self.eval()
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, hyper_param_dict=hyper_param_dict)
            if callback is not None:
                callback(data_dict, fwd_dict)

    def pred_mod1_mod2_forward(self,
        data_dict: Mapping[str, torch.Tensor]
    ) -> Mapping[str, Any]:
        """Forward pass for predicting first data modality from second data modality.

        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        """
        mod1_input = data_dict["cells_1_transformed"]
        batch_indices = data_dict.get('batch_indices', None)
        mod1_features = self.mod1_encoder(mod1_input)
        mu = mod1_features[:, :self.emb_dim]
        mu = F.normalize(mu)
        
        if not self.use_decoder:
            return dict(latents=mu, reconstruction=torch.zeros(mod1_input.shape))

        mod2_reconstruct = self.decode_mod2(mu, None, None, None, None, batch_indices, True)
        fwd_dict = dict(
            latents=mu,
            reconstruction=mod2_reconstruct['mod2_reconstruct']
        )
        return fwd_dict

    def pred_mod2_mod1_forward(self,
        data_dict: Mapping[str, torch.Tensor],
    ) -> Mapping[str, Any]:
        """Forward pass for predicting second data modality from the data modality.

        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        """
        mod2_input = data_dict["cells_2_transformed"]
        batch_indices = data_dict.get('batch_indices', None)
        mod2_features = self.mod2_encoder(mod2_input)
        mu = mod2_features[:, :self.emb_dim]
        mu = F.normalize(mu)
        if not self.use_decoder:
            return dict(latents=mu, reconstruction=torch.zeros(mod2_input.shape))

        mod1_reconstruct = self.decode_mod1(mu, None, None, None, None, batch_indices, True)

        fwd_dict = dict(
            latents=mu,
            reconstruction=mod1_reconstruct['mod1_reconstruct'],
        )
        return fwd_dict


class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        output = (
            torch.distributions.Normal(0, 1)
            .sample(
                (shape if isinstance(shape, torch.Size) else torch.Size([shape]))
                + torch.Size([self._dim + 1])
            )
            .to(self.device)
        )

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2], device=self.device)
            )
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma
