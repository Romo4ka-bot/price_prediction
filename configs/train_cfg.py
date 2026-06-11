import os
import math
from datetime import datetime

from easydict import EasyDict
from configs.data_cfg import cfg as data_cfg
from configs.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = EasyDict()

cfg.seed = 0
cfg.exp_dir = os.path.join(ROOT_DIR, 'runs', 'train',
                           datetime.now().strftime("%d-%m_%H-%M"))
cfg.batch_size = 1024  # 2048 1024
cfg.num_epoch = 600

cfg.wu_ratio = 0.05  # 0.1
cfg.decay_ratio = 0.85  # 0.75
cfg.lr = 6e-4 * cfg.batch_size / 256  # math.sqrt()  # 6e-4
cfg.lr_decay_factor = 1e-2
cfg.lr_decay = 'linear'  # cosine linear

cfg.optim = 'AdamW'
cfg.optim_args = {}
cfg.weight_decay = 3e-4

cfg.loss = 'L1Loss'  # SmoothL1Loss L1Loss MSELoss LogCosh HuberLoss
cfg.loss_args = {}  # 'delta': 8}  # 'reduction': 'batchmean'}

cfg.accelerator_args = {}  # 'mixed_precision': 'fp16'}  # , 'cpu': True}

# cfg.load_checkpoint = os.path.join(ROOT_DIR, 'runs', 'train', '15-02_13-08', "checkpoint")

cfg.task = 'train'
model_cfg.pred_dim = 1
data_cfg.include_target = model_cfg.mask_first_token
data_cfg.task = cfg.task

cfg.model_cfg = model_cfg
cfg.data_cfg = data_cfg

cfg.model = 'TablePredictor'

