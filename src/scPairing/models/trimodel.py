import logging
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

# import models.constants as constants
from ..models import constants
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .distributions import PowerSpherical, _kl_powerspherical_uniform
from .log_likelihood import log_nb_positive
from .losses import ClipLoss, DebiasedClipLoss, SigmoidLoss
from .utils import (
    ConcentrationEncoder,
    HypersphericalUniform,
    get_fully_connected_layers,
)

Loss = Literal['clip', 'debiased_clip', 'sigmoid']
Modalities = Literal['rna', 'atac', 'protein', 'other']
Combine = Literal['dropout', 'average']
ModalityNumber = Literal['mod1', 'mod2', 'mod3']


_logger = logging.getLogger(__name__)


class Trimodel(nn.Module):
    """Variational autoencoder model

    Parameters
    ----------
    mod1_input_dim
        Input dimension of the low-dimensional representations for the first data modality.
    mod2_input_dim
        Input dimension of the low-dimensional representations for the second data modality.
    mod3_input_dim
        Input dimension of the low-dimensional representations for the third data modality.
    mod1_var_dim
        Dimension of the count features for the first data modality.
    mod2_var_dim
        Dimension of the count features for the second data modality.
    mod3_var_dim
        Dimension of the count features for the third data modality.
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
    mod3_type
        The modality type of adata3. The options are identical to mod1_type.
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
        * ``'sigmoid'`` - sigmoid loss from `Sigmoid Loss for Language Image Pre-Training` (Zhai `et al.`, 2023)
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
    modality_discriminative
        Whether to adversarially train a discriminator that aims to
        predict which modality a particular embedding came from.
    batch_discriminative
        Whether to adversarially train a discriminator that aims to
        predict which batch a particular embedding came from.
    cosine_loss
        Whether to subtract the cosine similarity between the two modality embeddings
        into the total loss.
    set_temperature
        Temperature for the CLIP loss to use throughout training. If None, the temperature
        is learned.
    cap_temperature
        Maximum temperature the CLIP loss can reach during training. If None, the
        temperature is unbounded.
    """
    emb_names: Sequence[str] = [constants.MOD1_EMB, constants.MOD2_EMB, constants.MOD3_EMB]

    def __init__(
        self,
        mod1_input_dim: int,
        mod2_input_dim: int,
        mod3_input_dim: int,
        mod1_var_dim: int,
        mod2_var_dim: int,
        mod3_var_dim: int,
        n_batches: int = 1,
        mod1_type: Modalities = 'rna',
        mod2_type: Modalities = 'atac',
        mod3_type: Modalities = 'protein',
        use_decoder: bool = True,
        emb_dim: int = 10,
        encoder_hidden_dims: Sequence[int] = (128,),
        decoder_hidden_dims: Sequence[int] = (128,),
        reconstruct_mod1_fn: Optional[Callable] = None,
        reconstruct_mod2_fn: Optional[Callable] = None,
        reconstruct_mod3_fn: Optional[Callable] = None,
        loss_method: Loss = 'clip',
        combine_method: Combine = 'dropout',
        batch_dispersion: bool = False,
        use_norm: Literal['layer', 'batch', 'none'] = 'batch',
        dropout: float = 0.1,
        modality_discriminative: bool = True,
        batch_discriminative: bool = False,
        batch_discriminative_weight: float = 1,
        cosine_loss: bool = True,
        set_temperature: Optional[float] = None,
        cap_temperature: Optional[float] = None,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        super().__init__()

        # Validation and warnings
        # Validate modality types
        if mod1_type not in ['rna', 'atac', 'protein', 'other']:
            raise ValueError("mod1_type must be one of 'rna', 'atac', 'protein', or 'other'")
        if mod2_type not in ['rna', 'atac', 'protein', 'other']:
            raise ValueError("mod1_type must be one of 'rna', 'atac', 'protein', or 'other'")
        if mod3_type not in ['rna', 'atac', 'protein', 'other']:
            raise ValueError("mod1_type must be one of 'rna', 'atac', 'protein', or 'other'")
        # Validate batch choices
        if batch_dispersion and n_batches == 1:
            _logger.warning('With one batch provided, per-batch dispersion will be disabled')
            batch_dispersion = False
        if batch_discriminative and n_batches == 1:
            _logger.warning('With one batch provided, batch adversarial loss will be disabled')
            batch_discriminative = False
        # Validate decoder choices
        if (reconstruct_mod1_fn is not None or reconstruct_mod2_fn is not None) and not use_decoder:
            _logger.warning('Reconstruction functions were provided but decoding was turned off.\n'
                            'The provided reconstruction functions will not be used.')

        # Model parameters
        self.device: torch.device = device
        self.n_batches: int = n_batches
        self.need_batch: bool = n_batches > 1

        # Encoder input dimensions
        self.mod1_input_dim: int = mod1_input_dim
        self.mod2_input_dim: int = mod2_input_dim
        self.mod3_input_dim: int = mod3_input_dim
        # Decoder output dimensions
        self.mod1_output_dim: int = mod1_var_dim
        self.mod2_output_dim: int = mod2_var_dim
        self.mod3_output_dim: int = mod3_var_dim

        self.emb_dim: int = emb_dim
        self.encoder_hidden_dims: Sequence[int] = encoder_hidden_dims
        self.decoder_hidden_dims: Sequence[int] = decoder_hidden_dims
        self.norm: Literal['layer', 'batch', 'none'] = use_norm
        self.dropout: float = dropout

        self.mod1_type: Modalities = mod1_type
        self.mod2_type: Modalities = mod2_type
        self.mod3_type: Modalities = mod3_type

        # Encoder networks
        self.mod1_encoder: nn.Module = get_fully_connected_layers(
            self.mod1_input_dim,
            self.emb_dim,
            self.encoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod2_encoder: nn.Module = get_fully_connected_layers(
            self.mod2_input_dim,
            self.emb_dim,
            self.encoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod3_encoder: nn.Module = get_fully_connected_layers(
            self.mod3_input_dim,
            self.emb_dim,
            self.encoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.var_encoder: nn.Module = ConcentrationEncoder(
            self.mod1_input_dim,
            self.mod2_input_dim,
            self.mod3_input_dim
        )

        self.cap_temperature: Optional[float] = cap_temperature
        if set_temperature is not None:
            self.logit_scale: torch.Tensor = torch.ones([]) * set_temperature 
        else:
            self.logit_scale: nn.Parameter = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.modality_discriminative: bool = modality_discriminative
        if self.modality_discriminative:
            self.modality_discriminator = nn.Sequential(
                nn.Linear(self.emb_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 3)  # Directly predict binary
            )
            self.ce_loss: nn.Module = nn.CrossEntropyLoss()
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

        self.cosine_loss: bool = cosine_loss
        if self.cosine_loss:
            self.cosine_loss: nn.Module = nn.CosineSimilarity()

        self.emb_indices: torch.Tensor = torch.arange(self.emb_dim, dtype=torch.float) + 1

        self.loss_method: Loss = loss_method
        if loss_method == 'clip':
            self.clip_loss = ClipLoss()
        elif loss_method == 'debiased_clip':
            self.clip_loss = DebiasedClipLoss()
        elif loss_method == 'sigmoid':
            self.clip_loss = SigmoidLoss()
            self.bias = nn.Parameter(torch.ones([]) * 0)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.use_decoder: bool = use_decoder
        self.combine_method: Combine = combine_method

        self.reconstruct_mod1_fn: Optional[Callable] = reconstruct_mod1_fn
        self.reconstruct_mod2_fn: Optional[Callable] = reconstruct_mod2_fn
        self.reconstruct_mod3_fn: Optional[Callable] = reconstruct_mod3_fn
        self.batch_dispersion: bool = batch_dispersion
        self._init_decoders()

        self.to(device)

    def _init_decoders(self):
        """Initialize the decoders"""
        self.mod1_decoder: nn.Module = get_fully_connected_layers(
            self.emb_dim,
            self.mod1_input_dim,
            self.decoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod2_decoder: nn.Module = get_fully_connected_layers(
            self.emb_dim,
            self.mod2_input_dim,
            self.decoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod3_decoder: nn.Module = get_fully_connected_layers(
            self.emb_dim,
            self.mod3_input_dim,
            self.decoder_hidden_dims,
            norm_type=self.norm,
            dropout_prob=self.dropout
        )
        self.mod1_reconstructor = self.mod2_reconstructor = self.mod3_reconstructor = None
        if self.use_decoder and self.reconstruct_mod1_fn is None:
            mod1_decoder_input_dim = self.mod1_input_dim + self.n_batches if self.reconstruct_mod1_fn is None and self.n_batches > 1 else self.mod1_input_dim
            self.mod1_reconstructor: nn.Module = get_fully_connected_layers(
                mod1_decoder_input_dim,
                self.mod1_output_dim,
                self.decoder_hidden_dims,
                norm_type=self.norm,
                dropout_prob=self.dropout
            )
        if self.use_decoder and self.reconstruct_mod2_fn is None:
            mod2_decoder_input_dim = self.mod2_input_dim + self.n_batches if self.reconstruct_mod2_fn is None and self.n_batches > 1 else self.mod2_input_dim
            self.mod2_reconstructor: nn.Module = get_fully_connected_layers(
                mod2_decoder_input_dim,
                self.mod2_output_dim,
                self.decoder_hidden_dims,
                norm_type=self.norm,
                dropout_prob=self.dropout
            )
        if self.use_decoder and self.reconstruct_mod3_fn is None:
            mod3_decoder_input_dim = self.mod3_input_dim + self.n_batches if self.reconstruct_mod3_fn is None and self.n_batches > 1 else self.mod3_input_dim
            self.mod3_reconstructor: nn.Module = get_fully_connected_layers(
                mod3_decoder_input_dim,
                self.mod3_output_dim,
                self.decoder_hidden_dims,
                norm_type=self.norm,
                dropout_prob=self.dropout
            )

        if self.reconstruct_mod1_fn is None and (self.mod1_type == 'rna' or self.mod1_type == 'protein'):
            if self.batch_dispersion:
                self.mod1_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_batches, self.mod1_output_dim))
            else:
                self.mod1_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.mod1_output_dim))
        if self.reconstruct_mod2_fn is None and (self.mod2_type == 'rna' or self.mod2_type == 'protein'):
            if self.batch_dispersion:
                self.mod2_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_batches, self.mod2_output_dim))
            else:
                self.mod2_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.mod2_output_dim))
        if self.reconstruct_mod3_fn is None and (self.mod3_type == 'rna' or self.mod3_type == 'protein'):
            if self.batch_dispersion:
                self.mod3_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.n_batches, self.mod3_output_dim))
            else:
                self.mod3_dispersion: nn.Parameter = nn.Parameter(torch.rand(self.mod3_output_dim))

    def decode(
        self,
        decode_modality: ModalityNumber,
        features: torch.Tensor,
        embs: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        cell_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        is_imputation: bool = False
    ) -> Mapping[str, Any]:
        """Decode either the first modality or second modality data from the combined features

        Parameters
        ----------
        decode_modality
            Which modality to decode.
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
        if decode_modality == 'mod1':
            decoder = self.mod1_decoder
        elif decode_modality == 'mod2':
            decoder = self.mod2_decoder
        elif decode_modality == 'mod3':
            decoder = self.mod3_decoder

        approx_features = decoder(features)
        loss = torch.nn.MSELoss()(approx_features, embs) if not is_imputation else torch.zeros([])
        if not self.use_decoder:
            return {
                constants.FEATURES: approx_features,
                constants.RECONSTRUCTION: torch.zeros([]),
                constants.RECONSTRUCTION_LOSS: torch.zeros([]),
                constants.NLL: loss
            }
        if decode_modality == 'mod1':
            reconstruct_fn, reconstructor, mod_type, n_output = self.reconstruct_mod1_fn, self.mod1_reconstructor, self.mod1_type, self.mod1_output_dim
        elif decode_modality == 'mod2':
            reconstruct_fn, reconstructor, mod_type, n_output = self.reconstruct_mod2_fn, self.mod2_reconstructor, self.mod2_type, self.mod2_output_dim
        elif decode_modality == 'mod3':
            reconstruct_fn, reconstructor, mod_type, n_output = self.reconstruct_mod3_fn, self.mod3_reconstructor, self.mod3_type, self.mod3_output_dim
        if reconstruct_fn is None:
            if self.n_batches > 1:
                batch_one_hot = F.one_hot(batch_indices, num_classes=self.n_batches)
                if is_imputation:
                    batch_one_hot = torch.zeros(features.shape[0], self.n_batches)
                approx_features = torch.cat([approx_features, batch_one_hot], axis=1)
            approx_features = reconstructor(approx_features)
            if mod_type == 'rna' or mod_type == 'protein':
                if decode_modality == 'mod1':
                    dispersion = self.mod1_dispersion
                elif decode_modality == 'mod2':
                    dispersion = self.mod2_dispersion
                elif decode_modality == 'mod3':
                    dispersion = self.mod3_dispersion
                if is_imputation:
                    library_size = torch.ones([])
                reconstruct = F.softmax(approx_features, dim=1) * library_size
                dispersion = dispersion[batch_indices] if self.batch_dispersion else dispersion
                reconstruction_loss = -log_nb_positive(
                    counts,
                    reconstruct,
                    dispersion.exp()
                ).mean() / n_output if not is_imputation else torch.zeros([])
            elif mod_type == 'atac':
                reconstruct = torch.sigmoid(approx_features)
                reconstruction_loss = F.binary_cross_entropy(
                    reconstruct,
                    counts,
                    reduction="none"
                ).sum(-1).mean() * 10 / n_output if not is_imputation else torch.zeros([])
            elif mod_type == 'other':
                reconstruct = approx_features
                reconstruction_loss = torch.nn.MSELoss()(counts, reconstruct).mean() / n_output if not is_imputation else torch.zeros([])
            loss += reconstruction_loss
        else:
            reconstruct, _ = reconstruct_fn(
                approx_features,
                embs,
                counts,
                library_size,
                cell_indices,
                self.training,
                is_imputation,
                batch_indices
            )
            reconstruction_loss = torch.zeros([])
        return {
            constants.FEATURES: approx_features,
            constants.RECONSTRUCTION: reconstruct,
            constants.RECONSTRUCTION_LOSS: reconstruction_loss,
            constants.NLL: loss
        }

    def combine_features(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        mod3_features: torch.Tensor
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
            return (mod1_features + mod2_features + mod3_features) / 2
        elif self.combine_method == 'dropout':
            indices = torch.randint(0, 3, (self.emb_dim,))
            mod1_features[:, indices != 0] = 0
            mod2_features[:, indices != 1] = 0
            mod3_features[:, indices != 2] = 0
            combined = mod1_features + mod2_features + mod3_features
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
        counts_1, counts_2, counts_3 = data_dict["cells_1"], data_dict["cells_2"], data_dict["cells_3"]
        library_size_1, library_size_2, library_size_3 = data_dict["library_size_1"], data_dict["library_size_2"], data_dict["library_size_3"]
        cell_indices = data_dict['cell_indices']
        batch_indices = data_dict.get('batch_indices', None)

        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]
        mod3_input = data_dict["cells_3_transformed"]

        mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)
        mod3_features = self.mod3_encoder(mod3_input)

        mod1_features = F.normalize(mod1_features)
        mod2_features = F.normalize(mod2_features)
        mod3_features = F.normalize(mod3_features)

        combined_features = self.combine_features(mod1_features.clone(), mod2_features.clone(), mod3_features.clone())  # (batch_size * emb_dim)
        combined_features = combined_features / torch.norm(combined_features, p=2, dim=-1, keepdim=True)
        var = self.var_encoder(mod1_input, mod2_input, mod3_input) + constants.EPS  # (batch_size,)

        z_dist = PowerSpherical(combined_features, var.squeeze(-1))  # (batch_size, emb_dim)
        if self.training:
            z = z_dist.rsample()
        else:
            z = combined_features

        mod1_dict = self.decode('mod1', z, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
        mod2_dict = self.decode('mod2', z, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
        mod3_dict = self.decode('mod3', z, mod3_input, counts_3, library_size_3, cell_indices, batch_indices)

        log_px_zl = mod1_dict[constants.NLL] + mod2_dict[constants.NLL] + mod3_dict[constants.NLL]
        loss1 = mod1_dict[constants.NLL]
        loss2 = mod2_dict[constants.NLL]
        loss3 = mod3_dict[constants.NLL]

        logit_scale = self.logit_scale.exp()
        if self.cap_temperature is not None:
            logit_scale = torch.clamp(logit_scale, max=self.cap_temperature)

        fwd_dict = {
            constants.MOD1_EMB: mod1_features,
            constants.MOD2_EMB: mod2_features,
            constants.MOD3_EMB: mod3_features,
            constants.TEMP: logit_scale.mean(),
            constants.NLL: log_px_zl.mean(),
            constants.MOD1_RECONSTRUCT_LOSS: mod1_dict[constants.RECONSTRUCTION_LOSS],
            constants.MOD2_RECONSTRUCT_LOSS: mod2_dict[constants.RECONSTRUCTION_LOSS],
            constants.MOD3_RECONSTRUCT_LOSS: mod3_dict[constants.RECONSTRUCTION_LOSS],
            constants.MOD1_LOSS: loss1,
            constants.MOD2_LOSS: loss2,
            constants.MOD3_LOSS: loss3
        }

        if self.use_decoder:
            fwd_dict[constants.MOD1_RECONSTRUCT] = mod1_dict[constants.RECONSTRUCTION]
            fwd_dict[constants.MOD2_RECONSTRUCT] = mod2_dict[constants.RECONSTRUCTION]
            fwd_dict[constants.MOD3_RECONSTRUCT] = mod3_dict[constants.RECONSTRUCTION]

        if not self.training:
            return fwd_dict

        if self.loss_method == 'sigmoid':
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale, self.bias) + self.clip_loss(mod2_features, mod3_features, logit_scale, self.bias)
            fwd_dict['bias'] = self.bias
        elif self.loss_method == 'batch_clip':
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale, batch_indices) + \
                self.clip_loss(mod2_features, mod3_features, logit_scale, batch_indices) + \
                self.clip_loss(mod1_features, mod3_features, logit_scale, batch_indices)
        else:
            contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale) + \
                self.clip_loss(mod2_features, mod3_features, logit_scale) + \
                self.clip_loss(mod1_features, mod3_features, logit_scale)
        fwd_dict[constants.CONTRASTIVE] = contrastive_loss

        loss = log_px_zl + contrastive_loss

        if self.modality_discriminative and self.training:
            mod1_preds = self.modality_discriminator(mod1_features)
            mod2_preds = self.modality_discriminator(mod2_features)
            mod3_preds = self.modality_discriminator(mod3_features)
            truth = torch.cat([
                torch.zeros(mod1_features.shape[0], device=self.device),
                torch.ones(mod2_features.shape[0], device=self.device),
                2 * torch.ones(mod3_features.shape[0], device=self.device)]
            ).to(torch.long)  # [0, ..., 0, 1, ..., 1, 2, ..., 2]
            discriminative_loss = self.ce_loss(torch.cat([mod1_preds, mod2_preds, mod3_preds]).squeeze(-1), truth.squeeze(-1))
            fwd_dict['modality_discriminative'] = discriminative_loss

            loss = loss - discriminative_loss

        if self.batch_discriminative and self.training:
            batch_pred_1 = self.batch_discriminator(mod1_features)
            batch_pred_2 = self.batch_discriminator(mod2_features)
            batch_pred_3 = self.batch_discriminator(mod3_features)
            truth = torch.cat([batch_indices, batch_indices, batch_indices])
            batch_loss = self.ce_loss(torch.cat([batch_pred_1, batch_pred_2, batch_pred_3]).squeeze(-1), truth.squeeze(-1))
            fwd_dict['batch_discriminative'] = batch_loss

            loss = loss - batch_loss * hyper_param_dict.get('batch_weight', 1)

        if self.cosine_loss and self.training:
            dist_loss = self.cosine_loss(mod1_features, mod2_features).mean() + \
                self.cosine_loss(mod1_features, mod3_features).mean() + \
                self.cosine_loss(mod2_features, mod3_features).mean()
            # Want to encourage them to be similar
            loss = loss - dist_loss

            fwd_dict['dist'] = dist_loss

        uni = HypersphericalUniform(dim=self.emb_dim - 1, device=self.device)
        kl = _kl_powerspherical_uniform(z_dist, uni)
        fwd_dict[constants.KL] = kl.mean()
        loss += kl.mean() * hyper_param_dict.get('kl_weight', 1)

        record = {
            'loss': loss,
            constants.CONTRASTIVE: fwd_dict[constants.TEMP],
            constants.KL: fwd_dict[constants.KL],
            constants.MOD1_LOSS: fwd_dict[constants.MOD1_LOSS],
            constants.MOD2_LOSS: fwd_dict[constants.MOD2_LOSS],
            constants.MOD3_LOSS: fwd_dict[constants.MOD3_LOSS],
            constants.TEMP: fwd_dict[constants.TEMP]
        }
        if self.modality_discriminative and self.training:
            record['modality_discriminative'] = fwd_dict['modality_discriminative']
        if self.batch_discriminative and self.training:
            record['batch_discriminative'] = fwd_dict['batch_discriminative']
        if self.cosine_loss and self.training:
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
        batch_indices = data_dict.get('batch_indices', None)

        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]
        mod3_input = data_dict["cells_3_transformed"]

        mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)
        mod3_features = self.mod3_encoder(mod3_input)

        # L2 normalization
        mod1_features = F.normalize(mod1_features)
        mod2_features = F.normalize(mod2_features)
        mod3_features = F.normalize(mod3_features)

        discriminative_loss = 0
        if self.modality_discriminative:
            mod1_preds = self.modality_discriminator(mod1_features)
            mod2_preds = self.modality_discriminator(mod2_features)
            mod3_preds = self.modality_discriminator(mod3_features)
            truth = torch.cat([
                torch.zeros(mod1_features.shape[0], device=self.device),
                torch.ones(mod2_features.shape[0], device=self.device),
                2 * torch.ones(mod3_features.shape[0], device=self.device)]
            ).to(torch.long)  # [0, ..., 0, 1, ..., 1, 2, ..., 2]
            discriminative_loss = self.ce_loss(torch.cat([mod1_preds, mod2_preds, mod3_preds]).squeeze(-1), truth.squeeze(-1))

        batch_loss = 0
        if self.batch_discriminative:
            batch_pred_1 = self.batch_discriminator(mod1_features)
            batch_pred_2 = self.batch_discriminator(mod2_features)
            batch_pred_3 = self.batch_discriminator(mod3_features)
            truth = torch.cat([batch_indices, batch_indices, batch_indices])
            batch_loss = self.ce_loss(torch.cat([batch_pred_1, batch_pred_2, batch_pred_3]).squeeze(-1), truth.squeeze(-1))

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

    def intermodality_prediction(
        self,
        data_dict: Mapping[str, torch.Tensor],
        from_modality: ModalityNumber,
        to_modality: ModalityNumber
    ) -> Mapping[str, Any]:
        """Forward pass for predicting from one modality to another.
        
        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        from_modality
            The modality given to the model.
        to_modality
            The modality the model will predict.
        """
        mod_input = data_dict["cells_1_transformed"]
        batch_indices = data_dict.get('batch_indices', None)

        if from_modality == 'mod1':
            encoder = self.mod1_encoder
        elif from_modality == 'mod2':
            encoder = self.mod2_encoder
        elif from_modality == 'mod3':
            encoder = self.mod3_encoder
        
        features = encoder(mod_input)
        mu = F.normalize(features)

        reconstruct = self.decode(to_modality, mu, None, None, None, None, batch_indices, True)
        if not self.use_decoder:
            return {
                constants.REPRESENTATIONS: mu,
                constants.RECONSTRUCTION: reconstruct[constants.FEATURES]
            }
        return {
            constants.REPRESENTATIONS: mu,
            constants.RECONSTRUCTION: reconstruct[constants.RECONSTRUCTION]
        }
