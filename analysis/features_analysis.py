import os
import copy

import torch
import pandas as pd

from executors.trainer import Trainer
from configs.train_cfg import cfg
from utils.project_paths import get_dataset_dir, get_transformer_artifact_dir, load_tablepredictor_cfg


def data_analysis():
    path = get_dataset_dir()
    train = pd.read_csv(os.path.join(path, 'train.csv')).drop(columns=['Unnamed: 0'])
    print(train.info())
    # valid = pd.read_csv(os.path.join(path, 'valid.csv')).drop(columns=['Unnamed: 0'])
    # test = pd.read_csv(os.path.join(path, 'test.csv')).drop(columns=['Unnamed: 0'])
    # t = len(train) + len(valid) + len(test)
    # print(t)
    # print(len(train) / t)
    # print(len(valid) / t)
    # print(len(test) / t)


def feature_analysis():
    dt = cfg.data_cfg.data_transformer

    # t = dt.cat_processor.categories_.copy()
    # for i, (col, cats) in enumerate(zip(dt.cat_cols, dt.cat_processor.categories_)):
    #     t[i][pd.isna(cats)] = 'other'
    #
    for col, cats in zip(dt.cat_cols, dt.cat_processor.categories_):
        print(col, '\tКатегориальный\t', len(cats))

    for col, edges in zip(dt.num_cols, dt.num_processor.bin_edges_):
        print(col, '\tНепрерывный\t', len(edges))


@torch.no_grad()
def main():
    path = get_transformer_artifact_dir()
    cfg.model = 'TablePredictor'
    cfg.model_cfg = load_tablepredictor_cfg(path)
    trainer = Trainer(cfg, False)
    # w1 = copy.deepcopy(trainer.model.blocks[0].attn.qkv_proj.weight)
    w1 = copy.deepcopy(trainer.model.kv_compressors[0][1].weight)

    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    # w2 = copy.deepcopy(trainer.model.blocks[0].attn.qkv_proj.weight)
    w2 = copy.deepcopy(trainer.model.kv_compressors[0][1].weight)
    model = trainer.model

    print(torch.abs(w2 - w1).mean())
    # print(w1)
    # print(w2)
    model(torch.ones(1, 19, dtype=torch.long, device=next(model.parameters()).device))

    # print('qkv', torch.all(model.blocks[0].attn.qkv_proj.weight).item())
    # print('k', torch.all(model.kv_compressors[0][0].weight).item())
    # print('v', torch.all(model.kv_compressors[0][1].weight).item())
    # print('o', torch.all(model.blocks[0].attn.out_proj.weight == 0).item())


if __name__ == '__main__':
    main()
    # data_analysis()
    # feature_analysis()
