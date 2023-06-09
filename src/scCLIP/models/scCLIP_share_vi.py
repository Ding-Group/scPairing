from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Callable
import math
import anndata
import numpy as np
import logging
from scipy.sparse import spmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent

from logging_utils import log_arguments
from .BaseCellModel import BaseCellModel
from batch_sampler import CellSampler
from .log_likelihood import log_nb_positive
from .distributions import PowerSpherical


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
        lin = nn.Linear(input_dim, size)
        nn.init.xavier_normal_(lin.weight)
        layers.append(lin)
        layers.append(nn.ReLU())
        if bn:
            layers.append(nn.BatchNorm1d(size, track_running_stats=bn_track_running_stats))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ClipLoss(nn.Module):
    """
    Implementation of ClipLoss borrowed from OpenCLIP
    https://github.com/mlfoundations/open_clip
    """
    def __init__(
        self,
        cache_labels: bool = False
    ) -> None:
        """
        Initialize the CLIP loss
        """
        super().__init__()
        self.cache_labels = cache_labels
        
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
        device = atac_features.device
        logits_per_atac, logits_per_gex = self.get_logits(atac_features, gex_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_atac.shape[0])

        total_loss = (F.cross_entropy(logits_per_atac, labels) + F.cross_entropy(logits_per_gex, labels)) / 2

        return total_loss


class scCLIP(BaseCellModel):
    """

    """
    clustering_input: str = 'mod1_features'
    emb_names: Sequence[str] = ['mod1_features', 'mod2_features']

    def __init__(
        self,
        n_mod1: int,
        n_mod2: int,
        n_batches: int,
        emb_dim: int = 64,
        hidden_dims: Sequence[int] = (128,),
        bn: bool = True,
        dropout_prob: float = 0.1,
        decode_features: bool = True,
        combine_method: str = "dropout",
        linear_decoder: bool = False,
        loss_method: str = 'clip',
        cell_dropout: bool = False,
        cell_dropout_prob: float = 0.2,
        library_nn: bool = False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        """
        Initializes a CLIP with n_mod1 features for the first modality, and n_mod2
        features for the second modality in the dataset.
        The CLIP model will embed onto a hypersphere of dimension <emb_dim>.

        In the future, the two encoders should be passed in.
        """
        super().__init__(n_mod1, n_batches, need_batch=n_batches > 1, device=device)
        self.n_mod1 = n_mod1
        self.n_mod2 = n_mod2
        self.emb_dim = emb_dim
        self.hidden_dims = hidden_dims
        self.bn = bn
        self.dropout = dropout_prob
        self.beta = 7  # TODO: Make this a hyperparameter

        # Encoder networks
        self.cell_dropout = nn.Dropout(cell_dropout_prob) if cell_dropout else None
        # self.mod1_encoder = nn.Sequential(
        #     nn.Linear(self.n_mod1, 128),
        #     nn.LayerNorm(128),
        #     nn.ELU(),
        #     nn.Dropout(self.dropout),
        # )
        # self.mod2_encoder = nn.Sequential(
        #     nn.Linear(self.n_mod2, 128),
        #     nn.LayerNorm(128),
        #     nn.ELU(),
        #     nn.Dropout(self.dropout),
        # )
        self.mod1_encoder = get_fully_connected_layers(
            self.n_mod1,
            self.emb_dim,
            self.hidden_dims,
            bn=self.bn,
            dropout_prob=self.dropout
        )
        self.mod2_encoder = get_fully_connected_layers(
            self.n_mod2,
            self.emb_dim,
            self.hidden_dims,
            bn=self.bn,
            dropout_prob=self.dropout
        )
        self.mean_encoder = nn.Linear(self.emb_dim, self.emb_dim)
        self.var_encoder = nn.Linear(self.emb_dim, 1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Loss function
        self.loss_method = loss_method
        if self.loss_method == 'clip':
            self.loss = ClipLoss()  # Custom loss function
        elif self.loss_method == 'cosine':
            self.loss = torch.nn.CosineEmbeddingLoss()
        elif self.loss_method == 'euclidean':
            self.loss = torch.nn.MSELoss()

        self.decode_features = decode_features
        self.combine_method = combine_method
        self.linear_decoder = linear_decoder
        if self.decode_features:
            self._init_decoders()

        self.library_nn = library_nn
        if self.library_nn:
            self.library_size_nn_1 = get_fully_connected_layers(self.n_mod1 + 1, 1, (128,))
            # self.library_size_nn_2 = get_fully_connected_layers(self.n_mod1 + 1, 1, (128,))
        else:
            self.library_size_nn_1 = self.library_size_nn_2 = None

        self.to(device)

    def _init_decoders(self):
        if self.linear_decoder:
            if self.combine_method == 'concat':
                self.mod1_decoder = nn.Sequential(nn.Linear(self.emb_dim * 2, self.n_mod1), bias=False)
                self.mod2_decoder = nn.Sequential(nn.Linear(self.emb_dim * 2, self.n_mod2), bias=False)
            elif self.combine_method == 'average':
                self.mod1_decoder = nn.Sequential(nn.Linear(self.emb_dim, self.n_mod1), bias=False)
                self.mod2_decoder = nn.Sequential(nn.Linear(self.emb_dim, self.n_mod2), bias=False)
            elif self.combine_method == 'dropout':
                self.emb_indices = torch.arange(self.emb_dim, dtype=torch.float) + 1
                self.dropout_layer = torch.nn.Dropout()
                self.mod1_decoder = nn.Sequential(nn.Linear(self.emb_dim, self.n_mod1), bias=False)
                self.mod2_decoder = nn.Sequential(nn.Linear(self.emb_dim, self.n_mod2), bias=False)
        else:
            if self.combine_method == 'concat':
                self.mod1_decoder = get_fully_connected_layers(self.emb_dim * 2, self.n_mod1, self.hidden_dims[::-1])
                self.mod2_decoder = get_fully_connected_layers(self.emb_dim * 2, self.n_mod2, self.hidden_dims[::-1])
            elif self.combine_method == 'average':
                self.mod1_decoder = get_fully_connected_layers(self.emb_dim, self.n_mod1, self.hidden_dims[::-1])
                self.mod2_decoder = get_fully_connected_layers(self.emb_dim, self.n_mod2, self.hidden_dims[::-1])
            elif self.combine_method == 'dropout':
                self.emb_indices = torch.arange(self.emb_dim, dtype=torch.float)
                self.dropout_layer = torch.nn.Dropout()
                self.mod1_decoder = nn.Sequential(
                    nn.Linear(self.emb_dim, 128),
                    nn.LayerNorm(128),
                    nn.ELU(),
                    nn.Linear(128, self.n_mod1),
                    nn.ReLU()
                )
                self.mod2_decoder = nn.Sequential(
                    nn.Linear(self.emb_dim, 128),
                    nn.LayerNorm(128),
                    nn.ELU(),
                    nn.Linear(128, self.n_mod2),
                    nn.Sigmoid()  # Directly predict binary
                )
        self.mod1_dispersion = nn.Parameter(torch.rand(self.n_mod1))

    def decode(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor,
        counts_1: torch.Tensor,
        counts_2: torch.Tensor,
        library_size_1: torch.Tensor,
        library_size_2: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        Decode the features according to the decoding method.
        The decoding methods should compute the NLL.
        """
        # if self.combine_method == 'four-way':
        #     return self._decode_four_way(mod1_features, mod2_features, mod1_input, mod2_input, batch_indices)
        if self.combine_method == 'concat':
            return self._decode_concat(mod1_features, mod2_features, mod1_input, mod2_input, batch_indices)
        elif self.combine_method == 'average':
            return self._decode_average(mod1_features, mod2_features, mod1_input, mod2_input, batch_indices)
        elif self.combine_method == 'dropout':
            return self._decode_dropout(
                mod1_features.clone(),
                mod2_features.clone(),
                counts_1,
                counts_2,
                library_size_1,
                library_size_2,
                batch_indices
            )

    def new_decode(
        self,
        combined_features: torch.Tensor,
        counts_1: torch.Tensor,
        counts_2: torch.Tensor,
        library_size_1: torch.Tensor,
        library_size_2: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        New decode that takes in the combined features
        """
        mod1_reconstruct = self.mod1_decoder(combined_features)
        mod2_reconstruct = self.mod2_decoder(combined_features)

        # Adjust RNA counts for library size
        if self.library_size_nn_1:
            x_library_size = torch.cat([counts_1, library_size_1], -1)
            library_size_1 = self.library_size_nn_1(x_library_size).exp()
        mod1_reconstruct_means = mod1_reconstruct * library_size_1

        # Don't need to adjust ATAC for libary size as they are binarized

        # Calculate negative binomial loss
        mod1_dispersion = self.mod1_dispersion.exp()  # Take exponent to avoid <= 0 dispersions

        # Since this is a probability, need to take negative for loss.
        mod1_log_px = log_nb_positive(counts_1, mod1_reconstruct_means, mod1_dispersion)

        # F.binary_cross_entropy parameters
        # - input: tensor of arbitrary shape as probabilities
        # - target: tensor of the same shape as input with values between 0 and 1
        # Since this is a loss function, don't need to take its negative
        mod2_log_px = F.binary_cross_entropy(mod2_reconstruct, counts_2, reduction="none").sum(-1)
        # mod2_log_px = (-mod2_reconstruct * counts_2).sum(-1)

        return {
            'mod1_reconstruct': mod1_reconstruct_means,
            'mod2_reconstruct': mod2_reconstruct,
            'nll': mod1_log_px / self.n_mod1 - self.beta * mod2_log_px / self.n_mod2
        }


    def _decode_four_way(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        Decode the features into the two modality data using the four-way
        method from BABEL.
        """
        mod1_to_mod1 = F.log_softmax(mod1_features @ self.mod1_to_mod1, dim=-1)
        mod1_to_mod2 = F.log_softmax(mod1_features @ self.mod1_to_mod2, dim=-1)
        mod2_to_mod1 = F.log_softmax(mod2_features @ self.mod2_to_mod1, dim=-1)
        mod2_to_mod2 = F.log_softmax(mod2_features @ self.mod2_to_mod2, dim=-1)
        nll = (-mod1_to_mod1 * mod1_input).sum(-1).mean() + (-mod1_to_mod2 * mod2_input).sum(-1).mean() + \
              (-mod2_to_mod1 * mod1_input).sum(-1).mean() + (-mod2_to_mod2 * mod2_input).sum(-1).mean()
        return {
            'mod1_to_mod1': mod1_to_mod1,
            'mod1_to_mod2': mod1_to_mod2,
            'mod2_to_mod1': mod2_to_mod1,
            'mod2_to_mod2': mod2_to_mod2,
            'nll': nll / 4
        }
    
    def _decode_concat(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        Decode the features into the two modality data using a concatenation
        of the latent features.
        """
        concat = torch.cat([mod1_features, mod2_features], dim=1)
        mod1_reconstruct = F.log_softmax(self.mod1_decoder(concat), dim=-1)
        mod2_reconstruct = F.log_softmax(self.mod2_decoder(concat), dim=-1)
        nll = (-mod1_reconstruct * mod1_input).sum(-1).mean() + (-mod2_reconstruct * mod2_input).sum(-1).mean()
        return {
            'mod1_reconstruct': mod1_reconstruct,
            'mod2_reconstruct': mod2_reconstruct,
            'nll': nll / 2
        }
    
    def _decode_average(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        Decode the features into the two modality data using an average across
        the two latent features.
        """
        # Should this try to find the actual midpoint on a line betwen the two poitns
        # on the hypersphere?
        avg = (mod1_features + mod2_features) / 2
        mod1_reconstruct = F.log_softmax(self.mod1_decoder(avg), dim=-1)
        mod2_reconstruct = F.log_softmax(self.mod2_decoder(avg), dim=-1)
        nll = (-mod1_reconstruct * mod1_input).sum(-1).mean() + (-mod2_reconstruct * mod2_input).sum(-1).mean()
        return {
            'mod1_reconstruct': mod1_reconstruct,
            'mod2_reconstruct': mod2_reconstruct,
            'nll': nll / 2
        }
    
    def _decode_dropout(
        self,
        mod1_features: torch.Tensor,
        mod2_features: torch.Tensor,
        counts_1: torch.Tensor,
        counts_2: torch.Tensor,
        library_size_1: torch.Tensor,
        library_size_2: torch.Tensor,
        batch_indices: Union[None, torch.Tensor] = None) -> Mapping[str, Any]:
        """
        Decode the features into the two modality data using a combination
        of the two features with dropout.
        """
        mod1_indices = self.dropout_layer(self.emb_indices)
        mod1_indices = mod1_indices.long()
        mod1_features[:, mod1_indices == 0] = 0
        mod2_features[:, mod1_indices != 0] = 0
        combined = mod1_features + mod2_features
        # mod1_reconstruct = F.log_softmax(combined @ self.mod1_decoder, dim=-1)
        # mod2_reconstruct = F.log_softmax(combined @ self.mod2_decoder, dim=-1)

        # To implement negative binomial loss, need to have mean and dispersion neural network.

        mod1_reconstruct = self.mod1_decoder(combined)
        mod2_reconstruct = self.mod2_decoder(combined)

        if self.library_size_nn_1 is not None:
            mod1_count_lib = torch.cat([counts_1, library_size_1], -1)
            mod2_count_lib = torch.cat([counts_2, library_size_2], -1)
            library_size_1 = self.library_size_nn_1(mod1_count_lib).exp()
            library_size_2 = self.library_size_nn_2(mod2_count_lib).exp()

        mod1_reconstruct = mod1_reconstruct * library_size_1
        mod2_reconstruct = mod2_reconstruct * library_size_2

        # This won't be the nll later.
        nll = (-mod1_reconstruct * counts_1).sum(-1).mean() + (-mod2_reconstruct * counts_2).sum(-1).mean()
        return {
            'mod1_reconstruct': mod1_reconstruct,
            'mod2_reconstruct': mod2_reconstruct,
            'nll': nll / 2
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
            mod1_indices = self.dropout_layer(self.emb_indices) / 2  # Undo the correction that nn.Dropout does
            mod1_indices = mod1_indices.long()
            mod1_features[:, mod1_indices == 0] = 0
            mod2_features[:, mod1_indices != 0] = 0
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

        mod1_input = data_dict["cells_1_transformed"]
        mod2_input = data_dict["cells_2_transformed"]

        if self.cell_dropout:
            mod1_features = self.mod1_encoder(self.cell_dropout(mod1_input))
        else:
            mod1_features = self.mod1_encoder(mod1_input)
        mod2_features = self.mod2_encoder(mod2_input)

        # L2 normalization
        mod1_features = F.normalize(mod1_features)
        mod2_features = F.normalize(mod2_features)

        combined_features = self.combine_features(mod1_features.clone(), mod2_features.clone())

        mu = self.mean_encoder(combined_features)
        mu = mu / torch.norm(mu, p=2, dim=-1, keepdim=True)
        var = self.var_encoder(combined_features)
        var = nn.Softplus()(var)
    
        z_dist = PowerSpherical(mu, var.squeeze(-1))
        z = z_dist.rsample()

        if self.decode_features and hyper_param_dict.get("reconstruct_weight", 1):
            # If decoding is turned on and the reconstruction holds any weight in this procedure.
            decode_dict = self.new_decode(
                z,
                counts_1,
                counts_2,
                library_size_1,
                library_size_2
            )
            # decode_dict = self.decode(
            #     mod1_features,
            #     mod2_features,
            #     counts_1,
            #     counts_2,
            #     library_size_1,
            #     library_size_2,
            #     mod1_input,
            #     mod2_input
            # )
            log_px_zl = decode_dict['nll'] * hyper_param_dict.get("reconstruct_weight", 1)
        else:
            log_px_zl = 0
        
        fwd_dict = {
            "mod1_features": mod1_features,
            "mod2_features": mod2_features,
            "logit_scale": self.logit_scale.exp(),
            "log_px_zl": log_px_zl.mean()
        }
        if self.decode_features and hyper_param_dict.get("reconstruct_weight", 1):
            fwd_dict = fwd_dict | decode_dict

        fwd_dict['nll'] = decode_dict['nll'].mean()

        if not self.training:
            return fwd_dict

        if self.loss_method == 'clip' and hyper_param_dict.get("contrastive_weight", 1):
            contrastive_loss = self.loss(mod1_features, mod2_features, self.logit_scale) * hyper_param_dict.get("contrastive_weight", 1)
            fwd_dict['contrastive'] = contrastive_loss
            # print("CONTRASTIVE", loss - log_px_zl)
        elif self.loss_method == 'clip':
            contrastive_loss = 0
        elif self.loss_method == 'cosine':
            contrastive_loss = self.loss(mod1_features, mod2_features, torch.ones((mod1_features.shape[0],)))
        else:
            contrastive_loss = self.loss(mod1_features, mod2_features)

        log_pz = HypersphericalUniform(dim=self.emb_dim - 1, device=self.device).log_prob(z)
        log_qz_x = PowerSpherical(mu, var.squeeze(-1)).log_prob(z)
        log_ratio = log_px_zl + log_pz - log_qz_x
        obj = -log_ratio.mean(0)
        fwd_dict['obj'] = obj

        # print('log_px_zl', fwd_dict['log_px_zl'])
        # print('log_pz', log_pz.mean())
        # print('log_qz_x', log_qz_x.mean())
        # print('contrastive', contrastive_loss)
        loss = obj + contrastive_loss

        record = dict(loss=loss)
        record = {k: v.detach().item() for k, v in record.items()}
        return loss, fwd_dict, record
    
    def get_cell_embeddings_and_nll(
        self,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        batch_size: int = 2000,
        emb_names: Union[str, Iterable[str], None] = None,
        batch_col: str = 'batch_indices',
        inplace: bool = True
    ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:
        """
        Calculates cell embeddings
        """
        nlls = []
        if self.need_batch and adata_1.obs[batch_col].nunique() != self.n_batches:
            _logger.warning(
                f'adata.obs[{batch_col}] contains {adata_1.obs[batch_col].nunique()} batches, '
                f'while self.n_batches == {self.n_batches}.'
            )
            if self.need_batch:
                _logger.warning('Disable decoding. You will not get reconstructed cell-gene matrix or nll.')
                nlls = None
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
                embs[name].append(fwd_dict[name].detach().cpu())
            if nlls is not None:
                nlls.append(fwd_dict['nll'].detach().item())

        self._apply_to(adata_1, adata_2, batch_col, batch_size, hyper_param_dict, callback=store_emb_and_nll)

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
        hyper_param_dict: Union[dict, None] = None,
        callback: Union[Callable, None] = None
    ) -> None:
        """Docstring (TODO)
        """
        sampler = CellSampler(adata_1, adata_2, batch_size=batch_size, sample_batch_id=self.need_batch, n_epochs=1, batch_col=batch_col, shuffle=False)
        self.eval()
        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, hyper_param_dict=hyper_param_dict)
            if callback is not None:
                callback(data_dict, fwd_dict)

    def predict_mod1_to_mod2_forward(
        self,
        data_dict: Mapping[str, torch.Tensor],
        hyper_param_dict: Mapping[str, Any] = dict()
    ) -> Mapping[str, Any]:
        """
        Given data from modality 1, predict the corresponding
        expression of modality 2.
        """
        counts_1 = data_dict["cells_1"]
        library_size_1 = data_dict["library_size_1"]
        mod1_input = counts_1 / library_size_1
        mod1_features = self.mod1_encoder(mod1_input)

        mod1_to_mod2 = F.log_softmax(mod1_features @ self.mod1_to_mod2, dim=-1)

        fwd_dict = {
            "mod1_features": mod1_features,
            "log_mod2_predicted": mod1_to_mod2
        }


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
