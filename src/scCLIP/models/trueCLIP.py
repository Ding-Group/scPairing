from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Callable, Optional, List
import math
import anndata
import numpy as np
import logging
from scipy.sparse import spmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent

from logging_utils import log_arguments
from .BaseCellModel import BaseCellModel
from batch_sampler import CellSampler
from .log_likelihood import log_nb_positive
from .distributions import PowerSpherical, _kl_powerspherical_uniform


def get_fully_connected_layers(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    bn: bool = True,
    bn_track_running_stats: bool = True,
    dropout_prob: float = 0
) -> nn.Sequential:
    """
    Construct fully connected layers.
    
    Modified from scETM (https://github.com/hui2000ji/scETM)
    """
    layers = []
    for i, size in enumerate(hidden_dims):
        layers.append(nn.Linear(input_dim, size))
        layers.append(nn.ELU())
        if bn:
            layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


# def distance_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     Computes the total distance between x and y on the hypersphere
#     """
#     dot = torch.clamp(torch.sum(x * y, dim=-1), -1 + 1e-7, 1 - 1e-7)
#     return torch.mean(torch.acos(dot))


class ClipLoss(nn.Module):
    """
    Implementation of ClipLoss borrowed from OpenCLIP
    https://github.com/mlfoundations/open_clip
    """
    def __init__(
        self,
        cache_labels: bool = False,
        downsample_clip: bool = False,
        downsample_clip_prob: float = 0.5
    ) -> None:
        """
        Initialize the CLIP loss
        """
        super().__init__()
        self.cache_labels = cache_labels
        self.downsample_clip = downsample_clip
        self.downsample_clip_prob = downsample_clip_prob
        
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    def get_ground_truth(self, device, num_logits: int) -> torch.Tensor:
        """
        Get the ground truth. It is of form [0, 1, 2, ...], which means
        that each cell is of its own class.
        """
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, atac_features, gex_features, logit_scale):
        """
        Get the logits corresponding the unnormalized probability that
        the cell is actually itself.
        """
        logits_per_atac = logit_scale * atac_features @ gex_features.T
        logits_per_gex = logit_scale * gex_features @ atac_features.T
        return logits_per_atac, logits_per_gex

    def forward(self, atac_features, gex_features, logit_scale) -> float:
        """
        Compute forward Clip Loss. It is the average of the cross entropies computed
        from the two logits, logits_per_atac and logits_per_gex
        """
        if self.downsample_clip:
            mask = torch.rand(atac_features.shape[0]) < self.downsample_clip_prob
            atac_features = atac_features[mask, :]
            gex_features = gex_features[mask, :]
            if logit_scale.ndim > 0:
                logit_scale = logit_scale[mask]

        device = atac_features.device
        logits_per_atac, logits_per_gex = self.get_logits(atac_features, gex_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_atac.shape[0])

        total_loss = (F.cross_entropy(logits_per_atac, labels) + F.cross_entropy(logits_per_gex, labels)) / 2

        return total_loss


class DebiasedClipLoss(nn.Module):
    """
    Implementation of ClipLoss borrowed from OpenCLIP
    https://github.com/mlfoundations/open_clip
    """
    def __init__(
        self,
        tau: float = 0.3,
        cache_labels: bool = False,
        downsample_clip: bool = False,
        downsample_prob: float = 0.5
    ) -> None:
        """
        Initialize the CLIP loss
        """
        super().__init__()
        self.cache_labels = cache_labels
        self.tau = tau
        self.downsample_clip = downsample_clip
        self.downsample_prob = downsample_prob
        
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_logits(self, atac_features, gex_features, logit_scale):
        """
        Get the logits corresponding the unnormalized probability that
        the cell is actually itself.
        """
        logits_per_atac = logit_scale * atac_features @ gex_features.T
        logits_per_gex = logit_scale * gex_features @ atac_features.T
        return logits_per_atac, logits_per_gex

    def calc_loss(self, logits, logit_scale):
        logits = logits.exp()
        diag = logits.diagonal()
        N = logits.shape[0] - 1
        Ng = (-N * self.tau * diag + (torch.sum(logits, dim=-1) - diag)) / (1 - self.tau)
        b = N * (-1 / logit_scale).exp()
        Ng[Ng < b] = b
        return -torch.log(diag / (diag + Ng))

    def forward(self, atac_features: torch.Tensor, gex_features: torch.Tensor, logit_scale: torch.Tensor) -> float:
        """
        Compute forward Clip Loss. It is the average of the cross entropies computed
        from the two logits, logits_per_atac and logits_per_gex
        """
        if self.downsample_clip:
            mask = torch.rand(atac_features.shape[0]) < self.downsample_prob
            atac_features = atac_features[mask, :]
            gex_features = gex_features[mask, :]
            if logit_scale.ndim > 0:
                logit_scale = logit_scale[mask]

        logits_per_atac, logits_per_gex = self.get_logits(atac_features, gex_features, logit_scale)

        total_loss = (self.calc_loss(logits_per_atac, logit_scale) + self.calc_loss(logits_per_gex, logit_scale)) / 2

        return total_loss.mean()

# TODO: What about using the closest point embedded, like dictionary
# TODO: Simulate with just two discrete cell types
# TODO: What about supervised contrastive learning

class scCLIP(BaseCellModel):
    """

    """
    clustering_input: str = 'mod1_features'
    emb_names: Sequence[str] = ['mod1_features', 'mod2_features']

    def __init__(
        self,
        n_mod1_dim: int,
        n_mod2_dim: int,
        n_mod1_var: int,
        n_mod2_var: int,
        n_batches: int = 1,
        mod1_type: str = 'rna',
        mod2_type: str = 'atac',
        emb_dim: int = 64,
        hidden_dims: Sequence[int] = (128,),
        bn: bool = True,
        dropout_prob: float = 0.1,
        decode_features: bool = True,
        encode_hvar: bool = False,
        decode_hvar: bool = False,
        reconstruct_mod1_fn: Optional[Callable] = None,
        reconstruct_mod2_fn: Optional[Callable] = None,
        combine_method: str = "dropout",
        dropout_in_eval: bool = True,
        loss_method: str = 'clip',
        tau: float = 0.1,
        discriminative: bool = False,
        distance_loss: bool = False,
        variational: bool = False,
        downsample_clip: bool = False,
        downsample_clip_prob: float = 0.5,
        set_temperature: Optional[float] = None,
        cap_temperature: Optional[float] = None,
        per_cell_temperature: bool = False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """
        Initializes a CLIP with n_mod1 features for the first modality, and n_mod2
        features for the second modality in the dataset.
        The CLIP model will embed onto a hypersphere of dimension <emb_dim>.

        In the future, the two encoders should be passed in.
        """
        super().__init__(n_mod1_dim, n_batches, need_batch=n_batches > 1, device=device)

        self.n_mod1_input = n_mod1_dim
        self.n_mod2_input = n_mod2_dim
        self.n_mod1_output = n_mod1_var if reconstruct_mod1_fn is None else n_mod1_dim
        self.n_mod2_output = n_mod2_var if reconstruct_mod2_fn is None else n_mod2_dim
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.bn = bn
        self.dropout = dropout_prob
        self.encode_hvar = encode_hvar
        self.decode_hvar = decode_hvar
        self.mod1_type = mod1_type
        self.mod2_type = mod2_type

        # Encoder networks
        self.mod1_encoder = get_fully_connected_layers(
            self.n_mod1_input,
            self.emb_dim + 1,
            self.hidden_dims,
            bn=self.bn,
            dropout_prob=self.dropout
        )
        self.mod2_encoder = get_fully_connected_layers(
            self.n_mod2_input,
            self.emb_dim + 1,
            self.hidden_dims,
            bn=self.bn,
            dropout_prob=self.dropout
        )
        self.mean_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.var_encoder = nn.Linear(self.emb_dim, 1)

        self.per_cell_temperature = per_cell_temperature
        if set_temperature is not None:
            self.logit_scale = torch.ones([]) * set_temperature 
        elif per_cell_temperature:
            self.logit_scale_nn = nn.Sequential(
                nn.Linear(self.n_mod1_input + self.n_mod2_input, 16),
                nn.LayerNorm(16),
                nn.ELU(),
                nn.Linear(16, 1),
                nn.ReLU()
            )
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cap_temperature = cap_temperature

        self.discriminative = discriminative
        if self.discriminative:
            self.discriminator = nn.Sequential(
                nn.Linear(self.emb_dim, 128),
                nn.LayerNorm(128),
                nn.ELU(),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Directly predict binary
            )
            self.bce_loss = torch.nn.BCELoss()
        
        self.distance_loss = distance_loss
        if self.distance_loss:
            self.mse_loss = torch.nn.MSELoss()

        self.variational = variational
        if variational:
            self.dropout_layer = nn.Dropout()
            self.emb_indices = torch.arange(self.emb_dim, dtype=torch.float) + 1

        self.loss_method = loss_method
        if loss_method == 'clip':
            self.clip_loss = ClipLoss(downsample_clip=downsample_clip, downsample_clip_prob=downsample_clip_prob)
        elif loss_method == 'debiased':
            self.clip_loss = DebiasedClipLoss(tau, downsample_clip=downsample_clip, downsample_prob=downsample_clip_prob)
        else:
            raise ValueError("loss_method should be one of 'clip' or 'debiased'")

        self.decode_features = decode_features
        self.combine_method = combine_method
        self.dropout_in_eval = dropout_in_eval

        self.reconstruct_mod1_fn = reconstruct_mod1_fn
        if self.reconstruct_mod1_fn is None:
            self.mod1_dispersion = nn.Parameter(torch.rand(self.n_mod1_output))
        self.reconstruct_mod2_fn = reconstruct_mod2_fn
        if self.reconstruct_mod2_fn is None and self.mod2_type == 'protein':
            self.mod2_dispersion = nn.Parameter(torch.rand(self.n_mod2_output))
        if self.decode_features:
            self._init_decoders()

        self.device = device
        self.to(device)

    def _init_decoders(self):
        """
        Initialize the decoder from CLIP embedding to each model's dimension
        """
        self.mod1_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, self.n_mod1_output),
        )
        self.mod2_decoder = nn.Sequential(
            nn.Linear(self.emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Linear(256, self.n_mod2_output),
        )

    def decode_mod1(
        self,
        mod2_features: torch.Tensor,
        mod1_embs: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        cell_indices: torch.Tensor,
        batch_indices = None,
        is_imputation: bool = False,
    ) -> Mapping[str, Any]:
        """
        Decode first modality data from the mod2 features
        """
        approx_mod1_features = self.mod1_decoder(mod2_features)
        if self.reconstruct_mod1_fn is None:
            if is_imputation:
                library_size = torch.ones([])
            mod1_reconstruct = F.softmax(approx_mod1_features, dim=1) * library_size
            loss = -log_nb_positive(counts, mod1_reconstruct, self.mod1_dispersion.exp()).mean() / self.n_mod1_output if not is_imputation else None
        else:
            mod1_reconstruct, loss = self.reconstruct_mod1_fn(approx_mod1_features, mod1_embs, counts, library_size, cell_indices, self.training, is_imputation, batch_indices)
        return {
            'mod1_reconstruct': mod1_reconstruct,
            'nll': loss
        }

    def decode_mod2(
        self,
        mod1_features: torch.Tensor,
        mod2_embs: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor,
        cell_indices: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        is_imputation: bool = False,
    ) -> Mapping[str, Any]:
        """
        Decode the second modality data from the mod1 features
        """
        approx_mod2_features = self.mod2_decoder(mod1_features)
        if self.reconstruct_mod2_fn is None:
            if self.mod2_type == 'atac':
                mod2_reconstruct = torch.sigmoid(approx_mod2_features)
                loss = F.binary_cross_entropy(mod2_reconstruct, counts, reduction="none").sum(-1).mean() * 10 / self.n_mod2_output if not is_imputation else None
            else:
                mod2_reconstruct = F.relu(approx_mod2_features)
                loss = -log_nb_positive(counts, mod2_reconstruct, self.mod2_dispersion.exp()).mean() / self.n_mod2_output if not is_imputation else None
        else:
            mod2_reconstruct, loss = self.reconstruct_mod2_fn(approx_mod2_features, mod2_embs, counts, library_size, cell_indices, self.training, is_imputation, batch_indices)
        return {
            'mod2_reconstruct': mod2_reconstruct,
            'nll': loss
        }

    def combine_features(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor) -> torch.Tensor:
        """
        Combines the features using the combine_method of this model
        """
        if self.combine_method == 'concat':
            return torch.cat([mod1_features, mod2_features], dim=1)
        elif self.combine_method == 'average':
            averaged = (mod1_features + mod2_features) / 2
            return averaged / torch.norm(averaged, p=2, dim=-1, keepdim=True)
        elif self.combine_method == 'dropout':
            if self.training or not self.dropout_in_eval:
                mod1_indices = self.dropout_layer(self.emb_indices)
                mod1_indices = mod1_indices.long()
                mod1_features[:, mod1_indices == 0] = 0
                mod2_features[:, mod1_indices != 0] = 0
                combined = mod1_features + mod2_features
                return combined
            else:
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
        """
        Compute the forward pass
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

        if self.variational and self.decode_features:
            # combined_features = self.combine_features(mod1_features.clone(), mod2_features.clone())

            # mu = self.mean_encoder(combined_features)
            # mu = mu / torch.norm(mu, p=2, dim=-1, keepdim=True)
            # var = self.var_encoder(combined_features)  # Going to nan for some reason
            # var = nn.Softplus()(var) + 1e-5

            # z_dist = PowerSpherical(mu, var.squeeze(-1))
            # if self.training:
            #     z = z_dist.rsample()
            # else:
            #     z = mu
            if self.training:
                mod1_var = nn.Softplus()(mod1_var) + 1e-5
                mod2_var = nn.Softplus()(mod2_var) + 1e-5
                mod1_z_dist = PowerSpherical(mod1_features, mod1_var.squeeze(-1))
                mod2_z_dist = PowerSpherical(mod2_features, mod2_var.squeeze(-1))

                mod1_z = mod1_z_dist.rsample()
                mod2_z = mod2_z_dist.rsample()
            else:
                mod1_z, mod2_z = mod1_features, mod2_features
            
            z = self.combine_features(mod1_z.clone(), mod2_z.clone())
            z = F.normalize(z)

        if self.decode_features:
            if self.variational:
                mod1_dict = self.decode_mod1(z, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
                mod2_dict = self.decode_mod2(z, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
            else:
                mod1_dict = self.decode_mod1(mod2_features, mod1_input, counts_1, library_size_1, cell_indices, batch_indices)
                mod2_dict = self.decode_mod2(mod1_features, mod2_input, counts_2, library_size_2, cell_indices, batch_indices)
            log_px_zl = mod1_dict['nll'] + mod2_dict['nll']
            nb = mod1_dict['nll']
            bernoulli = mod2_dict['nll']
        else:
            log_px_zl = nb = bernoulli = torch.zeros([])

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
            "nb": nb,
            "bernoulli": bernoulli
        }
        
        if self.variational:
            fwd_dict['combined_features'] = z
        if self.decode_features:
            fwd_dict['mod1_reconstruct'] = mod1_dict['mod1_reconstruct']
            fwd_dict['mod2_reconstruct'] = mod2_dict['mod2_reconstruct']

        if not self.training:
            return fwd_dict


        contrastive_loss = self.clip_loss(mod1_features, mod2_features, logit_scale)
        fwd_dict['contrastive'] = contrastive_loss

        loss = log_px_zl + contrastive_loss  # TODO: is this -log_px_zl?
        
        if self.discriminative and self.training:
            mod1_preds = self.discriminator(mod1_features)
            mod2_preds = self.discriminator(mod2_features)
            truth = torch.cat([torch.zeros(mod1_features.shape[0], device=self.device), torch.ones(mod2_features.shape[0], device=self.device)])
            discriminative_loss = self.bce_loss(torch.cat([mod1_preds, mod2_preds]).squeeze(-1), truth.squeeze(-1))
            fwd_dict['discriminative'] = discriminative_loss

            loss = loss - discriminative_loss

        if self.distance_loss and self.training:
            dist_loss = self.mse_loss(mod1_features, mod2_features)
            loss = loss + dist_loss

            fwd_dict['dist'] = dist_loss

        if self.variational and self.decode_features:
            uni = HypersphericalUniform(dim=self.emb_dim - 1, device=self.device)
            # ps = PowerSpherical(mu, var.squeeze(-1))
            kl = _kl_powerspherical_uniform(mod1_z_dist, uni) + _kl_powerspherical_uniform(mod2_z_dist, uni)
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
        if self.discriminative and self.training:
            record['discriminative'] = fwd_dict['discriminative']
        if self.distance_loss and self.training:
            record['dist'] = fwd_dict['dist']
        record = {k: v.detach().item() for k, v in record.items()}
 
        # if self.discriminative and self.training:
            # return [loss, -discriminative_loss], fwd_dict, record
        return loss, fwd_dict, record

    def discriminative_forward(self, data_dict, hyper_param_dict):
        """
        Only get the discriminative loss
        """
        if not self.discriminative:
            return None
        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]

        mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)
        mod1_mu, _ = mod1_features[:, :self.emb_dim], mod1_features[:, self.emb_dim:]
        mod2_mu, _ = mod2_features[:, :self.emb_dim], mod2_features[:, self.emb_dim:]

        # L2 normalization
        mod1_features = F.normalize(mod1_mu)
        mod2_features = F.normalize(mod2_mu)
        # mod1_features = F.normalize(mod1_features)
        # mod2_features = F.normalize(mod2_features)

        mod1_preds = self.discriminator(mod1_features)
        mod2_preds = self.discriminator(mod2_features)
        truth = torch.cat([torch.zeros(mod1_features.shape[0], device=self.device), torch.ones(mod2_features.shape[0], device=self.device)])
        discriminative_loss = self.bce_loss(torch.cat([mod1_preds, mod2_preds]).squeeze(-1), truth.squeeze(-1))
        return discriminative_loss

    def get_cell_embeddings_and_nll(
        self,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        raw_layer=None,
        transformed_layer=None,
        transformed_obsm=None,
        inplace: bool = True
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """
        Calculates cell embeddings
        """
        nlls = []
        # if self.need_batch and adata_1.obs[batch_col].nunique() != self.n_batches:
            # _logger.warning(
            #     f'adata.obs[{batch_col}] contains {adata_1.obs[batch_col].nunique()} batches, '
            #     f'while self.n_batches == {self.n_batches}.'
            # )
            # if self.need_batch:
            #     _logger.warning('Disable decoding. You will not get reconstructed cell-gene matrix or nll.')
            #     nlls = None
        if not self.decode_features:
            nlls = None
        if emb_names is None:
            emb_names = self.emb_names
        self.eval()
        if isinstance(emb_names, str):
            emb_names = [emb_names]

        embs = {name: [] for name in emb_names}
        hyper_param_dict = dict(decode=nlls is not None, reconstruct_weight=1)

        def store_emb_and_nll(data_dict, fwd_dict):
            for name in emb_names:
                # print(len(embs[name]))
                embs[name].append(fwd_dict[name].detach().cpu())
            if nlls is not None:
                nlls.append(fwd_dict['nll'].detach().item())

        self._apply_to(
            adata_1, adata_2, batch_col, batch_size,
            raw_layer=raw_layer,
            transformed_layer=transformed_layer,
            transformed_obsm=transformed_obsm,
            hyper_param_dict=hyper_param_dict,
            callback=store_emb_and_nll
        )

        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}
        if nlls is not None:
            nll = sum(nlls) / adata_1.n_obs
        else:
            nll = None

        if inplace:
            adata_1.obsm.update(embs)
            adata_2.obsm.update(embs)
            return nll
        else:
            return embs, nll

    def _apply_to(self,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        batch_col: str = 'batch_indices',
        batch_size: int = 2000,
        raw_layer=None,
        transformed_layer=None,
        transformed_obsm=None,
        hyper_param_dict: Union[dict, None] = None,
        callback: Union[Callable, None] = None
    ) -> None:
        """Docstring (TODO)
        """
        sampler = CellSampler(
            adata_1, adata_2, use_highly_variable=self.encode_hvar,
            require_raw=self.decode_features,
            decode_highly_variable=self.decode_hvar,
            raw_layer=raw_layer,
            transformed_layer=transformed_layer,
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

    def _pred_mod1_mod2_forward(self,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any] = dict()
    ) -> Mapping[str, Any]:
        """
        Forward from mod1 to mod2
        """
        mod1_input = data_dict["cells_1_transformed"]
        batch_indices = data_dict.get('batch_indices', None)
        mod1_features = self.mod1_encoder(mod1_input)
        mod1_features = F.normalize(mod1_features)
        mu = mod1_features[:, :self.emb_dim]

        mod2_reconstruct = self.decode_mod2(mu, None, None, None, None, batch_indices, True)

        fwd_dict = dict(
            mod1_features=mod1_features,
            mu=mu,
            mod2_reconstruct=mod2_reconstruct['mod2_reconstruct']
        )
        return fwd_dict

    def _pred_mod2_mod1_forward(self,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any] = dict()
    ) -> Mapping[str, Any]:
        """
        Forward from mod2 to mod1
        """
        mod2_input = data_dict["cells_2_transformed"]
        batch_indices = data_dict.get('batch_indices', None)
        mod2_features = self.mod2_encoder(mod2_input)
        mod2_features = F.normalize(mod2_features)
        mu = mod2_features[:, :self.emb_dim]

        mod1_reconstruct = self.decode_mod1(mu, None, None, None, None, batch_indices, True)
        # mod2_reconstruct = self.mod2_decoder(mu)

        fwd_dict = dict(
            mod2_features=mod2_features,
            mu=mu,
            mod1_reconstruct=mod1_reconstruct['mod1_reconstruct'],
        )
        return fwd_dict

    def pred_mod1_to_mod2(self,
        adata_1: anndata.AnnData,
        transformed_layer: Optional[Union[str, List[str]]] = None,
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        batch_size: int = 2000,
        batch_col: str = 'batch_indices',
        inplace: bool = True,
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """
        Predict from modality 1 to modality 2.
        """
        hyper_param_dict = dict(decode=True, reconstruct_weight=1)
        sampler = CellSampler(
            adata_1,
            adata_1,  # We will ignore the second modality
            use_highly_variable=self.encode_hvar,
            decode_highly_variable=self.decode_hvar,
            transformed_layer=transformed_layer,
            transformed_obsm=transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.need_batch,
            n_epochs=1,
            batch_col=batch_col,
            shuffle=False
        )

        self.eval()
        embs = []
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self._pred_mod1_mod2_forward(data_dict, hyper_param_dict)
            embs.append(fwd_dict['mod2_reconstruct'].detach().cpu())
        
        # embs = {name: torch.cat(embs[name], dim=0).numpy() for name in ['mod1_pred', 'mod2_pred']}
        embs = torch.cat(embs, dim=0).numpy()

        if inplace:
            adata_1.obsm.update({'mod2_imputed': embs})
            return None
        else:
            return embs

    def pred_mod2_to_mod1(self,
        adata_2: anndata.AnnData,
        transformed_layer: Optional[Union[str, List[str]]] = None,
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        batch_size: int = 2000,
        batch_col: str = 'batch_indices',
        inplace: bool = True,
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """
        Predict from modality 2 to modality 1
        """
        hyper_param_dict = dict(decode=True, reconstruct_weight=1)
        sampler = CellSampler(
            adata_2,
            adata_2,  # We will ignore the second modality
            use_highly_variable=self.encode_hvar,
            decode_highly_variable=self.decode_hvar,
            transformed_layer=transformed_layer,
            transformed_obsm=transformed_obsm,
            batch_size=batch_size,
            sample_batch_id=self.need_batch,
            n_epochs=1,
            batch_col=batch_col,
            shuffle=False
        )

        self.eval()
        embs = []
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self._pred_mod2_mod1_forward(data_dict, hyper_param_dict)
            embs.append(fwd_dict['mod1_reconstruct'].detach().cpu())
        
        embs = torch.cat(embs, dim=0).numpy()

        if inplace:
            adata_2.obsm.update({'mod1_imputed': embs})
            return None
        else:
            return embs

    def predict_iterative(self,
        adata,
        transformed_obsm: str,  # something like X_pca
        embedding: str = 'mod1_features',
        num_iter: int = 100,
        batch_size: int = 10000,
        batch_col: str = 'batch_indices'
    ):
        """
        Predict mod2 features from mod1_features iteratively
        """
        mod1_features = torch.Tensor(adata.obsm[embedding])
        mod2_pred = torch.nn.Parameter(torch.rand(adata.obsm[transformed_obsm].shape))

        self.eval()
        optimizer = torch.optim.SGD([mod2_pred], lr=0.005)

        epoch = 0
        while epoch < num_iter:
            optimizer.zero_grad()
            mod2_features = F.normalize(self.mod2_encoder(mod2_pred))
            loss = -torch.sum(mod1_features * mod2_features, dim=-1).mean()
            loss.backward()
            optimizer.step()
            epoch += 1
        
        return mod2_pred.cpu().detach().numpy()


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
