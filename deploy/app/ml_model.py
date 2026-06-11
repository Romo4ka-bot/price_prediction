import os

import pandas as pd
import torch
import numpy as np
from typing import Dict, List
from easydict import EasyDict

from data.data_processing import DataTransformer
from deploy.app.schemas import PredictionInput
from models.transformers import TablePredictor
from utils.project_paths import get_transformer_artifact_dir, load_tablepredictor_cfg


class PricePredictor:
    """Класс для загрузки модели и предсказаний"""
    def __init__(self,
                 model_cfg: EasyDict,
                 model_path: str,
                 features_keys: List[str],
                 data_transformer: DataTransformer,
                 error_margin: float,
                 inflation: float
                 ) -> None:
        self.first_feature_idx = 1  # int(not model_cfg.mask_first_token)
        self.model = TablePredictor(**model_cfg)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.features_keys = features_keys
        self.data_transformer = data_transformer
        self.error_margin = error_margin
        self.inflation = inflation

    @torch.no_grad()
    def predict(self, data: PredictionInput) -> Dict:
        features = torch.as_tensor(
            self.data_transformer.transform(data.to_dataframe())
        )[:, self.first_feature_idx:]
        pred = self.model(features).numpy()
        price = self.data_transformer.inverse_transform(pred, target='num')
        price = self.inflation * price.item()
        lower_bound = price * (1 - self.error_margin)
        upper_bound = price * (1 + self.error_margin)
        res = {
            'predicted_price': int(price),
            'lower_bound': int(lower_bound),
            'upper_bound': int(upper_bound),
            'confidence': f"{(1 - self.error_margin) * 100:.0f}%",
            'error_margin': f"{self.error_margin * 100:.0f}%"
        }
        return res

    def get_feature_list(self) -> List[Dict]:
        features = []
        n_num_cols = len(self.data_transformer.num_cols)
        n_cat_cols = len(self.data_transformer.cat_cols)

        for i in range(1, n_num_cols):
            feature = dict(
                key=self.features_keys[i - 1],
                name=self.data_transformer.num_cols[i],
            )

            if 'площадь' in self.data_transformer.num_cols[i].lower():
                feature['unit'] = 'м²'
                feature['type'] = 'float'
            elif 'высота' in self.data_transformer.num_cols[i].lower():
                feature['unit'] = 'м'
                feature['type'] = 'float'
            else:
                feature['type'] = 'integer'

            if self.data_transformer.num_processor.bin_edges_[i][0] < 0:
                feature['required'] = False
                feature['min'] = int(self.data_transformer.num_processor.bin_edges_[i][2])
            else:
                feature['required'] = True
                feature['min'] = int(self.data_transformer.num_processor.bin_edges_[i][0])

            features.append(feature)

        for i in range(n_cat_cols):
            feature = dict(
                key=self.features_keys[n_num_cols + i - 1],
                name=self.data_transformer.cat_cols[i],
                type='select',
            )
            options = self.data_transformer.cat_processor.categories_[i]
            mask = pd.isna(options)

            if mask.any():
                options[mask] = 'Другое'
                feature['required'] = False
            else:
                feature['required'] = True

            feature['options'] = options.tolist()
            if i > n_cat_cols - 3:
                feature['options'].append('Другое')

            features.append(feature)

        return features


# Singleton instance
_predictor = None


def get_predictor() -> PricePredictor:
    """Возвращает singleton экземпляр предиктора"""
    from configs.data_cfg import cfg as data_cfg

    global _predictor

    if _predictor is None:
        path = get_transformer_artifact_dir()
        model_path = os.path.join(path, 'TablePredictor.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Не найдены веса модели: {model_path}. '
                'Укажите каталог модели через PRICE_PREDICTION_MODEL_DIR.'
            )

        model_cfg = load_tablepredictor_cfg(path)
        print(model_cfg)
        _predictor = PricePredictor(
            model_cfg=load_tablepredictor_cfg(path),
            model_path=model_path,
            features_keys=['area', 'living_area', 'kitchen_area', 'floor', 'total_floors',
                           'passenger_elevators', 'cargo_elevators', 'rooms', 'ceiling_height',
                           'separate_bathrooms', 'sale_type', 'object_type', 'garbage_chute',
                           'parking', 'house_type', 'window_view', 'distance_to_metro',
                           'district', 'neighborhood'],
            # features_keys=list(PredictionInput().dict().keys()),
            data_transformer=data_cfg.data_transformer,
            inflation=2.14,
            error_margin=0.042
        )
    return _predictor
