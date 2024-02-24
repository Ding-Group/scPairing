from typing import Any, Iterable, Mapping, Sequence, Tuple, Union, Optional, Callable, Literal, List
import numpy as np
import logging
from scipy.sparse import spmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


_logger = logging.getLogger(__name__)


def get_fully_connected_layers(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    norm_type: Literal['layer', 'batch', 'none'] = 'layer',
    dropout_prob: float = 0.1
) -> nn.Sequential:
    """Construct fully connected layers.
    """
    layers = []
    for i, size in enumerate(hidden_dims):
        lin = nn.Linear(input_dim, size)
        # nn.init.xavier_normal_(lin.weight)
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


class ConcentrationEncoder(nn.Module):
    """PyTorch module to compute the Power Spherical concentration
    parameter using both low-dimension representations.
    """
    def __init__(
        self,
        mod1_input_dim: int,
        mod2_input_dim: int,
        intermediate_dim: int = 5,
        norm_type: Literal['layer', 'batch', 'none'] = 'layer',
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()

        self.mod1_encoder = get_fully_connected_layers(
            mod1_input_dim,
            intermediate_dim,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob
        )
        self.mod2_encoder = get_fully_connected_layers(
            mod2_input_dim,
            intermediate_dim,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob
        )
        self.final = get_fully_connected_layers(
            intermediate_dim * 2,
            1,
            (32,),
            norm_type=norm_type,
            dropout_prob=dropout_prob
        )

    def forward(
        self,
        mod1_input: torch.Tensor,
        mod2_input: torch.Tensor
    ) -> None:
        """Compute the forward pass
        """
        x = torch.concat([self.mod1_encoder(mod1_input), self.mod2_encoder(mod2_input)], dim=1)
        return nn.Softplus()(self.final(x))
