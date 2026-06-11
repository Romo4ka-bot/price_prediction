import os
from datetime import datetime

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'runs', 'pretrain',
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
cfg.batch_size = 8 * 1024
cfg.num_epoch = 125

cfg.wu_ratio = 0.05
cfg.decay_ratio = 0.5
cfg.lr = 6e-4 * (cfg.batch_size / 256) ** 0.5  # 1e-3 bs: 8 * 1024
cfg.lr_decay_factor = 1e-2
cfg.lr_decay = 'cosine'

cfg.optim = 'AdamW'
cfg.optim_args = {'betas': (0.9, 0.95)}
cfg.weight_decay = 1e-5

cfg.loss = 'CrossEntropyLoss'  # CrossEntropyLoss SmoothL1Loss KLDivLoss L1Loss
cfg.loss_args = {}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {'mixed_precision': 'fp16', 'cpu': True}

cfg.model = 'MaskedTableAutoencoder'
cfg.mask_ratio = 0.65
model_cfg.decoder_embed_dim = model_cfg.embed_dim // 2
model_cfg.decoder_num_heads = model_cfg.num_heads
model_cfg.decoder_num_blocks = max(1, model_cfg.num_blocks // 3)

# cfg.model = 'MaskedTableModeling'
# cfg.mask_ratio = 0.5

cfg.task = 'pretrain'
model_cfg.pred_dim = sum(model_cfg.num_embed_features)
data_cfg.include_target = model_cfg.mask_first_token
data_cfg.task = cfg.task

cfg.data_cfg = data_cfg
cfg.model_cfg = model_cfg
