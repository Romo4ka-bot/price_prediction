import torch
import torch.nn as nn


class Compressor(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 seq_len: int,
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, 2*embed_dim)
        self.split_size = [embed_dim] + [embed_dim // 2] * 2
        self.comp_proj = nn.Linear(seq_len, 1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x).split(self.split_size, 2)
        x = x[1].transpose(1, 2)
        x = self.comp_proj(x).repeat_interleave(2, dim=1)
        x = self.out_proj(x.transpose(1, 2))
        return x


class GatedCompressor(nn.Module):
    def __init__(self,
                 seq_len: int,
                 dim_factor: float,
                 act: str,
                 dropout: float
                 ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(seq_len, 2 * round(dim_factor * seq_len))
        self.act = getattr(nn, act)()
        self.out_proj = nn.Linear(round(dim_factor * seq_len), 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(x) * y
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
