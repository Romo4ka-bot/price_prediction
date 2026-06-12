import json
import os
import time

from easydict import EasyDict

# from models.ensembles import PricePredEnsemble
import models.tabm as tabm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from data.apartment_dataset import ApartmentDataset
from utils.logger import Logger
from utils.utils import set_seed, get_scheduler, mape, get_param_groups, LogCoshLoss


class TabmTrainer:
    def __init__(self, cfg):
        set_seed(cfg.seed)

        self.logger = Logger(cfg)
        self.accelerator = Accelerator(**cfg.accelerator_args)
        self.cfg = cfg

        self.best_epoch = -1
        self.best_loss = float('inf')
        self.time_training = 0
        self.start_epoch = 1
        self.best_metric = 1e8

        self._prepare_data(cfg.data_cfg)
        self._prepare_model(cfg.model_cfg)

        self.logger.print('Training on ' + str(self.accelerator.device))

    def _prepare_data(self, data_cfg):
        self.data_transformer = data_cfg.data_transformer

        self.train_data = ApartmentDataset("train", **data_cfg)
        self.valid_data = ApartmentDataset('valid', **data_cfg)

        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 2
            kwargs['pin_memory'] = True
        self.train_dataloader = DataLoader(self.train_data, shuffle=True, **kwargs)
        self.val_dataloader = DataLoader(self.valid_data, shuffle=False, **kwargs)

    def _prepare_model(self, model_cfg):
        self.model = tabm.TabM.make(**model_cfg)
        self.criterion = LogCoshLoss(**self.cfg.loss_args) if self.cfg.loss == 'LogCoshLoss' \
            else getattr(nn, self.cfg.loss)(**self.cfg.loss_args)
        self.optimizer = getattr(torch.optim, self.cfg.optim)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            **self.cfg.optim_args
        )
        self.scheduler = get_scheduler(self.optimizer, len(self.train_dataloader) * self.cfg.num_epoch,
                                       self.cfg.lr_decay, self.cfg.lr, self.cfg.lr_decay_factor,
                                       self.cfg.wu_ratio, self.cfg.decay_ratio)
        (
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader,
            self.val_dataloader, self.scheduler
        )
        pretrained_path = self.cfg.get('load_pretrained', False)
        if pretrained_path:
            self.load_model(pretrained_path, strict=False)
            self.model.zero_compressors_()
            self.logger.print('load_pretrained')

        checkpoint_path = self.cfg.get('load_checkpoint', False)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            self.logger.print('load_checkpoint')

    def metric(self, pred, label):
        pred = pred.cpu()
        label = label.cpu()
        pred = torch.as_tensor(
            self.data_transformer.inverse_transform(pred, target='num')
        )
        return mape(pred, label)

    def save_model(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path, **kwargs)

    def load_model(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, f"{self.cfg.model}.pt")
        self.model.load_state_dict(
            torch.load(load_path, map_location=self.accelerator.device),
            **kwargs
        )

    def save_checkpoint(self, epoch, save_path=None, **kwargs):
        if save_path is None:
            save_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.save_state(save_path, **kwargs)
        with open(os.path.join(save_path, "extra_states.json"), "w") as f:
            json.dump(
                {
                    'epoch': epoch,
                    'best_epoch': self.best_epoch,
                    'best_loss': self.best_loss,
                    'best_metric': self.best_metric,
                    'time_training': self.time_training
                },
                f
            )

    def load_checkpoint(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = os.path.join(self.cfg.exp_dir, "checkpoint")
        self.accelerator.load_state(load_path, **kwargs)

        with open(os.path.join(load_path, "extra_states.json")) as f:
            extra_states = json.load(f)
            self.start_epoch = extra_states['epoch'] + 1
            self.best_epoch = extra_states['best_epoch']
            self.best_loss = extra_states['best_loss']
            self.best_metric = extra_states['best_metric']
            self.time_training = extra_states['time_training']

    def make_step(self, batch, update_model=True):
        with self.accelerator.autocast():
            pred = self.model(x_cat=batch['features']).squeeze()
            target = batch['target'].repeat(1, pred.size(1))
            # print(pred.shape)
            # print(target.shape)
            loss = self.criterion(pred, target) ** 0.5

        if update_model:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()

        pred = pred.mean(1)
        return loss.item() ** 2, pred.detach()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for batch in self.train_dataloader:
            loss, pred = self.make_step(batch)
            batch_len = len(pred)
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        self.time_training += t

        return {
            'loss': total_loss ** 0.5,
            'metric': total_metric,
            'time': t
        }

    @torch.no_grad()
    def evaluate(self,):
        self.model.eval()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for batch in self.val_dataloader:
            loss, pred = self.make_step(batch, False)
            batch_len = len(pred)
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        return {
            'loss': total_loss ** 0.5,
            'metric': total_metric,
            'time': t
        }

    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.num_epoch + 1):
            train = self.train_epoch()
            valid = self.evaluate()

            self.logger.log_metrics(epoch, train, valid)
            if valid['metric'] < self.best_metric:
                self.logger.print('Best')
                self.save_model()
                self.best_metric = valid['metric']
                self.best_loss = valid['loss']
                self.best_epoch = epoch
            else:
                self.logger.print(
                    f"Best | epoch: {self.best_epoch} | metric: "
                    f"{self.best_metric:.5f} | loss: {self.best_loss:.5f}"
                )
            self.save_checkpoint(epoch)
            self.logger.save_plot('loss')
            self.logger.save_plot('metric')
            # print(self.optimizer.param_groups[0]['lr'])

    def overfitting_on_batch(self, max_step=1000):
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 100 == 0:
                self.logger.print(f'[{step}]: loss - {loss:.4f}')

    @torch.no_grad()
    def test(self):
        kwargs = {'batch_size': self.cfg.batch_size}
        if torch.cuda.is_available():
            kwargs['num_workers'] = 2
            kwargs['pin_memory'] = True
        dataloader = DataLoader(
            ApartmentDataset("test", **self.cfg.data_cfg),
            shuffle=True, **kwargs
        )
        self.model, dataloader = self.accelerator.prepare(self.model, dataloader)

        self.model.eval()
        total_loss = 0
        total_metric = 0
        total_samples = 0

        t = time.time()
        for batch in dataloader:
            loss, pred = self.make_step(batch, False)
            batch_len = len(pred)
            total_samples += batch_len
            total_loss += loss * batch_len
            total_metric += self.metric(pred, batch['label']) * batch_len

        t = time.time() - t
        total_loss /= total_samples
        total_metric /= total_samples
        return {
            'loss': total_loss,
            'metric': total_metric,
            'time': t
        }


if __name__ == "__main__":
    from configs.train_cfg import cfg
    # from utils.project_paths import get_tabm_artifact_dir
    #
    # cfg.batch_size = 16
    # path = get_tabm_artifact_dir()
    # with open(os.path.join(path, 'logs', 'config.json'), 'r') as f:
    #     cfg.model_cfg = json.load(f)['model_cfg']
    #
    # trainer = TabmTrainer(cfg)
    # trainer.load_model(os.path.join(path, 'TabM.pt'))
    #
    # print(trainer.test())

    cfg.num_epoch = 200
    cfg.weight_decay = 3e-4
    cfg.lr = 2e-3
    cfg.model = 'TabM'
    cfg.batch_size = 64

    cfg.loss = 'MSELoss'

    cfg.model_cfg = EasyDict(
        cat_cardinalities=cfg.model_cfg.num_embed_features,
        d_out=1,
        # arch_type='tabm-mini'
        # d_in,
        # n_blocks,
        # d_block,
        # dropout=0.1,
        # activation='ReLU',
        # k=32
    )

    trainer = TabmTrainer(cfg)
    # trainer.overfitting_on_batch()
    trainer.fit()


"""
best 08-02_14-13
"""
