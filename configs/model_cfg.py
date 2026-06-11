from easydict import EasyDict

from configs.data_cfg import cfg as data_cfg

cfg = EasyDict()

cfg.num_q_heads = 16 # 8, 12, 24
cfg.num_kv_heads = 8 # 4, 6, 12
cfg.embed_dim = 10 * cfg.num_q_heads # 8, 16, 32
cfg.num_blocks = 40  # 32
cfg.act = 'ReLU'  # SiLU

cfg.pool = 'mean'  # mean token
# cfg.include_target = False  # False True
cfg.mask_first_token = False
cfg.add_first_token = cfg.pool == 'token' and not cfg.mask_first_token
cfg.num_embed_features = (
    data_cfg.data_transformer.num_bins[0 if cfg.mask_first_token else 1:] +
    data_cfg.data_transformer.num_cats
)

cfg.attn_dropout = 0.0
cfg.mlp_dropout = 0.1
cfg.dropout = 0.1
cfg.kv_compression_ratio = 0.01
cfg.mlp_dim_factor = 2

cfg.attn = 'Attention'  # Linear
cfg.mlp = 'GLUMLP'  # GLUMLP MLP
cfg.norm = 'RMSNorm'  # LayerNorm RMSNorm




