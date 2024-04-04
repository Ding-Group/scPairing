from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import math
import anndata
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from batch_sampler import CellSampler
from .log_likelihood import log_nb_positive
from .distributions import PowerSpherical, _kl_powerspherical_uniform
from .losses import ClipLoss, DebiasedClipLoss, SigmoidLoss
from .utils import get_fully_connected_layers, ConcentrationEncoder

Loss = Literal['clip', 'debiased_clip', 'sigmoid']
Modalities = Literal['rna', 'atac', 'protein', 'other']
Combine = Literal['dropout', 'average']


_logger = logging.getLogger(__name__)


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
    modality_discriminative
        Whether to adversarially train a discriminator that aims to
        predict which modality a particular embedding came from.
    batch_discriminative
        Whether to adversarially train a discriminator that aims to
        predict which batch a particular embedding came from.
    distance_loss
        Whether to subtract the cosine similarity between the two modality embeddings
        into the total loss.
    set_temperature
        Temperature for the CLIP loss to use throughout training. If None, the temperature
        is learned.
    cap_temperature
        Maximum temperature the CLIP loss can reach during training. If None, the
        temperature is unbounded.
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
        set_temperature: Optional[float] = None,
        cap_temperature: Optional[float] = None,
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

        # Model parameters
        self.device: torch.device = device
        self.n_batches: int = n_batches
        self.need_batch: bool = n_batches > 1

        # Encoder input dimensions
        self.mod1_input_dim: int = mod1_input_dim
        self.mod2_input_dim: int = mod2_input_dim
        # Decoder output dimensions
        self.mod1_output_dim: int = mod1_var_dim
        self.mod2_output_dim: int = mod2_var_dim

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
        self.var_encoder: nn.Module = ConcentrationEncoder(
            self.mod1_input_dim,
            self.mod2_input_dim
        )

        self.cap_temperature: Optional[float] = cap_temperature
        if set_temperature is not None:
            self.logit_scale: torch.Tensor = torch.ones([]) * set_temperature 
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
        self.mod1_reconstructor = self.mod2_reconstructor = None
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

    def decode(
        self,
        decode_mod1: bool,
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
        decode_mod1
            Whether to decode the first modality or the second modality
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
        decoder = self.mod1_decoder if decode_mod1 else self.mod2_decoder

        approx_features = decoder(features)
        loss = torch.nn.MSELoss()(approx_features, embs) if not is_imputation else torch.zeros([])
        if not self.use_decoder:
            return {
                'approx_features': approx_features,
                'reconstruction': torch.zeros([]),
                'reconstruction_loss': torch.zeros([]),
                'nll': loss
            }
        reconstruct_fn, reconstructor, mod_type, n_output = (self.reconstruct_mod1_fn, self.mod1_reconstructor, self.mod1_type, self.mod1_output_dim) \
            if decode_mod1 else (self.reconstruct_mod2_fn, self.mod2_reconstructor, self.mod2_type, self.mod2_output_dim)
        if reconstruct_fn is None:
            if self.n_batches > 1:
                if is_imputation:
                    batch_one_hot = torch.zeros((features.shape[0], self.n_batches), device=self.device)
                else:
                    batch_one_hot = F.one_hot(batch_indices, num_classes=self.n_batches)
                approx_features = torch.cat([approx_features, batch_one_hot], axis=1)
            approx_features = reconstructor(approx_features)
            if mod_type == 'rna':
                dispersion = self.mod1_dispersion if decode_mod1 else self.mod2_dispersion
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
            'approx_features': approx_features,
            'reconstruction': reconstruct,
            'reconstruction_loss': reconstruction_loss,
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
            mask = torch.rand(mod1_features.shape) < 0.5
            mod1_features[mask] = 0
            mod2_features[~mask] = 0
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
        counts_1, counts_2 = data_dict["cells_1"], data_dict["cells_2"]  # (batch_size, mod1_output_dim), (batch_size, mod2_output_dim)
        library_size_1, library_size_2 = data_dict["library_size_1"], data_dict["library_size_2"]  # (batch_size,), (batch_size,)
        cell_indices = data_dict['cell_indices']  # (batch_size,)
        batch_indices = data_dict.get('batch_indices', None)  # (batch_size,)

        mod1_input = data_dict["cells_1_transformed"] # (batch_size * mod1_input_dim)
        mod2_input = data_dict["cells_2_transformed"] # (batch_size * mod2_input_dim)

        mod1_features = F.normalize(self.mod1_encoder(mod1_input))  # (batch_size * emb_dim)
        mod2_features = F.normalize(self.mod2_encoder(mod2_input))  # (batch_size * emb_dim)

        combined_features = self.combine_features(mod1_features.clone(), mod2_features.clone())  # (batch_size * emb_dim)
        combined_features = combined_features / torch.norm(combined_features, p=2, dim=-1, keepdim=True)
        if self.variational:
            var = self.var_encoder(mod1_input, mod2_input) + 1e-5  # (batch_size,)

            z_dist = PowerSpherical(combined_features, var.squeeze(-1))  # (batch_size, emb_dim)
            if self.training:
                z = z_dist.rsample()
            else:
                z = combined_features
        else:
            z = combined_features

        if self.variational:
            mod1_dict = self.decode(True, z, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
            mod2_dict = self.decode(False, z, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
        else:
            mod1_dict = self.decode(True, mod2_features, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
            mod2_dict = self.decode(False, mod1_features, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
        log_px_zl = mod1_dict['nll'] + mod2_dict['nll']
        nb = mod1_dict['nll']
        bernoulli = mod2_dict['nll']

        logit_scale = self.logit_scale.exp()
        if self.cap_temperature is not None:
            logit_scale = torch.clamp(logit_scale, max=self.cap_temperature)

        fwd_dict = {
            "mod1_features": mod1_features,
            "mod2_features": mod2_features,
            "temp": logit_scale.mean(),
            "nll": log_px_zl.mean(),
            "mod1_reconstruct_loss": mod1_dict["reconstruction_loss"],
            "mod2_reconstruct_loss": mod2_dict["reconstruction_loss"],
            "nb": nb,
            "bernoulli": bernoulli
        }
        
        if self.variational and self.use_decoder:
            fwd_dict['combined_features'] = z
        if self.use_decoder:
            fwd_dict['mod1_reconstruct'] = mod1_dict['reconstruction']
            fwd_dict['mod2_reconstruct'] = mod2_dict['reconstruction']

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

        # L2 normalization
        mod1_features = F.normalize(mod1_features)
        mod2_features = F.normalize(mod2_features)

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
        mu = F.normalize(mod1_features)

        mod2_reconstruct = self.decode(False, mu, None, None, None, None, batch_indices, True)
        if not self.use_decoder:
            return dict(latents=mu, reconstruction=mod2_reconstruct['approx_features'])
        return dict(latents=mu, reconstruction=mod2_reconstruct['reconstruction'])

    def pred_mod2_mod1_forward(self,
        data_dict: Mapping[str, torch.Tensor],
    ) -> Mapping[str, Any]:
        """Forward pass for predicting second data modality from the data modality.

        Parameters
        ----------
        data_dict
            Dictionary containing the minibatch training data.
        """
        mod2_input = data_dict["cells_1_transformed"]  # Single modality data
        batch_indices = data_dict.get('batch_indices', None)
        mod2_features = self.mod2_encoder(mod2_input)
        mu = F.normalize(mod2_features)

        mod1_reconstruct = self.decode(True, mu, None, None, None, None, batch_indices, True)
        if not self.use_decoder:
            return dict(latents=mu, reconstruction=mod1_reconstruct['approx_features'])
        return dict(latents=mu, reconstruction=mod1_reconstruct['reconstruction'])


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
