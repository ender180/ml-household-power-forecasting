from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1])


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


class MultiScaleConvTransformerForecaster(nn.Module):
    """Improved model: local multi-scale convolutions before global Transformer encoding."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        branch_dim = d_model // 4
        self.input_proj = nn.Linear(input_dim, branch_dim)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(input_dim, branch_dim, kernel_size=k, padding=k // 2)
                for k in (3, 7, 15)
            ]
        )
        merged_dim = branch_dim * 4
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = [branch(x.transpose(1, 2)).transpose(1, 2) for branch in self.branches]
        z = torch.cat([self.input_proj(x), *local], dim=-1)
        z = self.pos(self.merge(z))
        z = self.encoder(z)
        weights = torch.softmax(self.attn_pool(z), dim=1)
        pooled = torch.sum(z * weights, dim=1)
        return self.head(pooled)


def build_model(name: str, input_dim: int, horizon: int, **kwargs: object) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMForecaster(input_dim=input_dim, horizon=horizon, **kwargs)
    if name == "transformer":
        return TransformerForecaster(input_dim=input_dim, horizon=horizon, **kwargs)
    if name in {"conv_transformer", "msct", "improved"}:
        return MultiScaleConvTransformerForecaster(input_dim=input_dim, horizon=horizon, **kwargs)
    raise ValueError(f"Unknown model: {name}")
