import os
import math
import torch
import torch.nn as nn
import numpy as np

from executors.trainer import Trainer
from configs.train_cfg import cfg
from utils.project_paths import get_transformer_artifact_dir, load_tablepredictor_cfg


@torch.no_grad()
def get_feature_importance(model):

    def kv_compressor_weights(compressor):
        if isinstance(compressor, nn.ModuleList):
            return sum([x.weight.abs().mean(dim=0).cpu().numpy() for x in compressor])
            # return sum([x.weight.norm(dim=0).cpu().numpy() for x in compressor])
        elif isinstance(compressor, nn.Linear):
            return compressor.weight.abs().mean(dim=0).cpu().numpy()
            # return compressor.weight.norm(dim=0).cpu().numpy()
        raise NotImplementedError

    if isinstance(model.kv_compressors, nn.ModuleList):
        w = np.sum(
            [kv_compressor_weights(compressor)
             for i, compressor in enumerate(model.kv_compressors)],
            axis=0
        )
    elif isinstance(model.kv_compressors, nn.Linear):
        w = kv_compressor_weights(model.kv_compressors)
    else:
        raise NotImplementedError

    ids = np.argsort(w)[::-1]
    features = (cfg.data_cfg.data_transformer.num_cols +
                cfg.data_cfg.data_transformer.cat_cols)
    print(w[ids])
    print(np.array(features[int(not cfg.data_cfg.include_target):])[ids])

@torch.no_grad()
def main():
    path = get_transformer_artifact_dir()
    cfg.model = 'TablePredictor'
    cfg.model_cfg = load_tablepredictor_cfg(path)
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    model = trainer.model

    def kv_compressor_weights(compressor):
        if isinstance(compressor, nn.ModuleList):
            return sum([x.weight.abs().sum(dim=0).cpu().numpy() for x in compressor])
        elif isinstance(compressor, nn.Linear):
            return compressor.weight.abs().sum(dim=0).cpu().numpy()
        raise NotImplementedError

    # if isinstance(model.kv_compressors, nn.ModuleList):
    #     w = np.sum(
    #         [
    #             kv_compressor_weights(compressor) / ((i + 1) ** 2)
    #             for i, compressor in enumerate(model.kv_compressors)
    #         ],
    #         axis=0
    #     )

    w1 = kv_compressor_weights(model.kv_compressors[0])
    for i in range(1, len(model.kv_compressors)):
        w2 = kv_compressor_weights(model.kv_compressors[i])
        print(w2 - w1)
        w1 = w2


if __name__ == '__main__':
    path = get_transformer_artifact_dir()
    cfg.model = 'TablePredictor'
    cfg.model_cfg = load_tablepredictor_cfg(path)
    trainer = Trainer(cfg, False)
    trainer.load_model(os.path.join(path, 'TablePredictor.pt'))
    # model = trainer.model
    # get_feature_importance()
    get_feature_importance_v2(trainer.model)


"""
[2.7843444  2.777239   2.2842464  2.1284864  2.0257008  1.9084042
 1.7282344  1.4424212  1.2407185  1.2059945  1.1965979  1.1403681
 1.068266   1.0404059  0.997353   0.96044624 0.9294794  0.9158576
 0.48160803]
['Общая площадь' 'Этаж' 'Район' 'Округ' 'Площадь кухни' 'Этажей в доме'
 'Жилая площадь' 'Расстояние до метро' 'Тип продажи' 'Вид из окон'
 'Высота потолков' 'Количество комнат' 'Кол-во раздельных санузлов'
 'Тип дома' 'Объект продажи' 'Лифт пассажирский (кол-во)' 'Парковка'
 'Лифт грузовой (кол-во)' 'Мусоропровод']
"""
