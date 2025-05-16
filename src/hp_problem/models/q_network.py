from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):
    """Dueling MLP that maps flattened grid (B,C,H,W) → Q-values.

    Parameters
    ----------
    in_dim : int  – flattened observation size
    n_actions : int
    hidden : tuple[int,int]  – two hidden-layer sizes
    """

    def __init__(self, in_dim: int, n_actions: int, hidden: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        # value & advantage streams
        self.val = nn.Linear(h2, 1)
        self.adv = nn.Linear(h2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, *)
        x = x.float().view(x.size(0), -1)
        feats = self.backbone(x)
        v = self.val(feats)                       # (B,1)
        a = self.adv(feats)                       # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)   # (B,A)
        return q


class NoisyLinear(nn.Module):
    """Factorized Noisy Linear layer (NoisyNet) for improved exploration.  

    Implements factorized Gaussian noise per Fortunato et al.  
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / self.in_features**0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features**0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features**0.5)

    def reset_noise(self) -> None:
        """Sample new noise for each forward pass."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class CNNDuelingQNet(nn.Module):
    """Dueling Q-network with CNN backbone and NoisyNet linear layers.  

    Args:
        board_size: int  # side length of square grid observation
        n_actions: int   # number of discrete actions
        hidden_dims: tuple[int,int] = (128, 128)  # sizes of hidden fully-connected layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        hidden_dims: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.n_actions = n_actions
        h1, h2 = hidden_dims

        # CNN feature extractor: input channels=3 (empty/H/P one-hot)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Noisy fully-connected dueling streams
        self.fc1 = NoisyLinear(64, h1)
        self.fc2 = NoisyLinear(h1, h2)
        self.val = NoisyLinear(h2, 1)
        self.adv = NoisyLinear(h2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W) int grid (0=empty,1=H,2=P)
        # one-hot encode into 3 channels
        x = x.long()
        x = F.one_hot(x, num_classes=3).permute(0, 3, 1, 2).float()
        feat = self.conv(x).view(x.size(0), -1)  # (B,64)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))

        v = self.val(feat)                # (B,1)
        a = self.adv(feat)                # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self) -> None:
        """Reset noise on all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
