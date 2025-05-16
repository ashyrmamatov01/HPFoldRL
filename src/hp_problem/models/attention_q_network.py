from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 10_000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        return x + self.pe[: x.size(1)]


class AttnDuelingQNet(nn.Module):
    """Dueling network with Transformer-encoder backbone.

    Args
    ----
    board_size : int
        side length of the square lattice
    n_actions : int
        number of discrete actions (3 for 2-D HP)
    d_model : int, default=128
        transformer model dimension
    n_head : int, default=4
        number of attention heads
    depth : int, default=2
        number of Transformer encoder layers
    """

    def __init__(
        self,
        board_size: int,
        n_actions: int,
        d_model: int = 128,
        n_head: int = 4,
        depth: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.seq_len = board_size * board_size
        self.token_dim = 3  # empty / H / P one-hot

        # Token projection
        self.embed = nn.Linear(self.token_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=self.seq_len)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Dueling heads
        self.val = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.adv = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_actions))

    # --------------------------------------------------------------------- #
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten board and one-hot encode (B, H, W) -> (B, L, C)."""
        x = x.long()                      # 0,1,2
        x = F.one_hot(x, num_classes=3)   # (B, H, W, 3)
        x = x.view(x.size(0), self.seq_len, self.token_dim).float()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W) -> (B, L, C)
        tok = self._prep(x)
        z = self.embed(tok)           # (B, L, D)
        z = self.pos_enc(z)
        z = self.transformer(z)       # (B, L, D)
        z = z.mean(dim=1)             # global mean-pool

        v = self.val(z)               # (B,1)
        a = self.adv(z)               # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
