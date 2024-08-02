import logging
import math
import random
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)


def get_fully_connected_layers(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    norm_type: Literal["layer", "batch", "none"] = "layer",
    dropout_prob: float = 0.1,
) -> nn.Sequential:
    """Construct fully connected layers."""
    layers = []
    for i, size in enumerate(hidden_dims):
        layers.append(nn.Linear(input_dim, size))
        layers.append(nn.ELU())
        if norm_type == "layer":
            layers.append(nn.LayerNorm(size))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm1d(size))
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ConcentrationEncoder(nn.Module):
    """PyTorch module to compute the Power Spherical concentration
    parameter using both low-dimension representations.
    """

    def __init__(
        self,
        mod1_input_dim: int,
        mod2_input_dim: int,
        mod3_input_dim: Optional[int] = None,
        intermediate_dim: int = 5,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        n_mods = 3 if mod3_input_dim else 2
        self.mod1_encoder = get_fully_connected_layers(
            mod1_input_dim,
            intermediate_dim,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob,
        )
        self.mod2_encoder = get_fully_connected_layers(
            mod2_input_dim,
            intermediate_dim,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob,
        )
        if mod3_input_dim:
            self.mod3_encoder = get_fully_connected_layers(
                mod3_input_dim,
                intermediate_dim,
                (32,),
                norm_type=norm_type,
                dropout_prob=dropout_prob,
            )
        self.final = get_fully_connected_layers(
            intermediate_dim * n_mods,
            1,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor,
        mod3_input: Optional[torch.Tensor] = None,
    ) -> None:
        """Compute the forward pass"""
        n1 = self.mod1_encoder(mod1_input)
        n2 = self.mod2_encoder(mod2_input)
        if mod3_input is None:
            x = torch.concat([n1, n2], dim=1)
        else:
            n3 = self.mod3_encoder(mod3_input)
            x = torch.concat([n1, n2, n3], dim=1)
        return nn.Softplus()(self.final(x))


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
    _logger.info(f'Set seed to {seed}.')
