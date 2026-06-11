import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.embedding import FeatureEmbedding
from models.modules.block import TransformerBlock
from models.modules.mlp import GatedMLP


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_q_heads: int,
                 num_kv_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 num_blocks: int,
                 attn: str,
                 mlp: str,
                 norm: str,
                 pool: str,
                 pred_dim: int,
                 add_cls_token: bool,
                 mask_first_token: bool,
                 kv_compression: str = None,
                 kv_compression_ratio: float = None,
                 ) -> None:
        super().__init__()
        self.add_cls_token = add_cls_token
        self.embed = FeatureEmbedding(num_embed_features, embed_dim,
                                      dropout, add_cls_token)
        self.seq_len = self.embed.seq_len

        self.kv_compression_dim = max(1, int(kv_compression_ratio * self.seq_len))

        self.kv_compressors = nn.ModuleList([
            nn.ModuleList([self._get_compressor(), self._get_compressor()])
            for _ in range(num_blocks)
        ])

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_q_heads, num_kv_heads, attn_dropout, mlp_dropout,
                             dropout, act, mlp_dim_factor, attn, mlp, norm)
            for _ in range(num_blocks)
        ])
        self.norm = getattr(nn, norm)(embed_dim)

    def _get_compressor(self) -> nn.Linear:
        return nn.Linear(
            self.seq_len,
            # 1,
            self.kv_compression_dim,
            False
        )

    def reset_parameters(self) -> None:
        for pn, p in self.named_parameters():
            if 'norm' not in pn:
                if 'bias' in pn:
                    nn.init.zeros_(p)
                elif 'head' in pn:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                else:
                    nn.init.normal_(p, std=0.02)

class TablePredictor(Transformer):
    def __init__(self,
                 embed_dim: int,
                 num_embed_features: list[int],
                 num_q_heads: int,
                 num_kv_heads: int,
                 attn_dropout: float,
                 mlp_dropout: float,
                 dropout: float,
                 act: str,
                 mlp_dim_factor: float,
                 num_blocks: int,
                 attn: str,
                 mlp: str,
                 norm: str,
                 pool: str,
                 pred_dim: int,
                 add_cls_token: bool,
                 mask_first_token: bool,
                 kv_compression: str = None,
                 kv_compression_ratio: float = None,
                 ) -> None:
        super().__init__(embed_dim, num_embed_features, num_q_heads, num_kv_heads, attn_dropout,
                         mlp_dropout, dropout, act, mlp_dim_factor, num_blocks, attn, mlp, norm,
                         pool, pred_dim, add_cls_token, mask_first_token, kv_compression,
                         kv_compression_ratio)
        self.pool = pool
        self.mask_first_token = mask_first_token
        if mask_first_token:
            self.register_buffer('mask', torch.zeros(1, self.seq_len, dtype=torch.bool))
            self.mask[:, 0] = True
        self.tp_head = nn.Linear(embed_dim, pred_dim)

        self.reset_parameters()
        if pool == 'w_avg':
            self.w_avg = nn.Parameter(torch.ones(self.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_first_token:
            x = self.embed(x, self.mask)
        else:
            x = self.embed(x)

        for i, block in enumerate(self.blocks):
            x = block(
                x,
                self.kv_compressors[i]
            )

        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        elif self.pool == 'sum':
            x = x.sum(1)
        elif self.pool == 'max':
            x = x.max(1).values
        elif self.pool == 'w_avg':
            x = self.w_avg.softmax(0) @ x
        else:
            raise NotImplementedError()

        x = self.norm(x)
        x = self.tp_head(x)
        return x

if __name__ == '__main__':
    from configs.model_cfg import cfg

    m = Transformer(**cfg, pred_dim=1)
    o = m.configure_optimizer(0.1, 0.1)
    for k, _ in m.named_parameters():
        print(k)