from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def get_logits(self, atac_features: torch.Tensor, gex_features: torch.Tensor, logit_scale) -> Tuple[torch.Tensor, torch.Tensor]:
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


class BatchClipLoss(ClipLoss):
    """
    CLIP loss applied to only a batch?
    """
    def __init__(self, weight: float):
        """
        l is the weight of 
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, atac_features: torch.Tensor, gex_features: torch.Tensor, logit_scale, batch_indices: torch.Tensor):
        device = atac_features.device
        logits_per_atac, logits_per_gex = self.get_logits(atac_features, gex_features, logit_scale)
        batch_equals = batch_indices[:, None] - batch_indices[None, :]
        batch_equals[batch_equals != 0] = self.weight
        batch_equals[batch_equals == 0] = 1

        logits_per_atac = logits_per_atac + batch_equals.log() - (logits_per_atac.exp() * batch_equals).sum(-1).log().unsqueeze(-1)
        logits_per_gex = logits_per_gex + batch_equals.log() - (logits_per_gex.exp() * batch_equals).sum(-1).log().unsqueeze(-1)

        labels = self.get_ground_truth(device, logits_per_atac.shape[0])

        return (F.nll_loss(logits_per_atac, labels) + F.nll_loss(logits_per_gex, labels)) / 2


class SigmoidLoss(nn.Module):
    """
    Implementation of Sigmoid loss
    """
    def __init__(self) -> None:
        super().__init__()
        self.log_sigmoid = torch.nn.LogSigmoid()

    def get_logits(self, atac_features, gex_features, logit_scale, bias):
        """
        Get the logits corresponding the unnormalized probability that
        the cell is actually itself.
        """
        logits = logit_scale * atac_features @ gex_features.T + bias
        return logits

    def forward(self, mod1_features, mod2_features, logit_scale, bias):
        n = mod1_features.shape[0]
        logits = self.get_logits(mod1_features, mod2_features, logit_scale, bias)
        labels =  2 * torch.eye(n, device=mod1_features.device) - torch.ones((n, n), device=mod1_features.device)
        return -self.log_sigmoid(labels * logits).sum() / n
