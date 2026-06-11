import copy
import json
import os
from typing import Any, Dict, Optional


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'datasets')
DEFAULT_TRANSFORMER_DIR = os.path.join(ROOT_DIR, 'веса и конфиги', 'transformer')
DEFAULT_TABM_DIR = os.path.join(ROOT_DIR, 'веса и конфиги', 'веса для tabm')
DEFAULT_CATBOOST_MODEL_PATH = os.path.join(ROOT_DIR, 'веса и конфиги', 'catboost.cbm')


def get_path(env_var: str, default_path: str) -> str:
    return os.environ.get(env_var, default_path)


def get_dataset_dir() -> str:
    return get_path('PRICE_PREDICTION_DATA_DIR', DEFAULT_DATASET_DIR)


def get_transformer_artifact_dir() -> str:
    return get_path('PRICE_PREDICTION_MODEL_DIR', DEFAULT_TRANSFORMER_DIR)


def get_tabm_artifact_dir() -> str:
    return get_path('PRICE_PREDICTION_TABM_DIR', DEFAULT_TABM_DIR)


def get_catboost_model_path() -> str:
    return get_path('PRICE_PREDICTION_CATBOOST_MODEL', DEFAULT_CATBOOST_MODEL_PATH)


def normalize_tablepredictor_cfg(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = copy.deepcopy(model_cfg)

    if 'add_first_token' in model_cfg:
        model_cfg['add_cls_token'] = model_cfg.pop('add_first_token')
    model_cfg.setdefault(
        'add_cls_token',
        model_cfg.get('pool') == 'cls' and not model_cfg.get('mask_first_token', False),
    )

    if 'kv_compression_ratio' in model_cfg:
        kv_compression_ratio = model_cfg.pop('kv_compression_ratio')
        seq_len = len(model_cfg.get('num_embed_features', [])) + int(model_cfg.get('add_cls_token', False))
        model_cfg.setdefault(
            'kv_compression_dim',
            max(1, int(seq_len * kv_compression_ratio)) if model_cfg.get('kv_compression') else None,
        )

    return model_cfg


def load_tablepredictor_cfg(model_dir: Optional[str] = None) -> Dict[str, Any]:
    model_dir = model_dir or get_transformer_artifact_dir()
    config_path = os.path.join(model_dir, 'logs', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f'Не найден config.json модели: {config_path}. '
            'Укажите каталог модели через PRICE_PREDICTION_MODEL_DIR.'
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        model_cfg = json.load(f)['model_cfg']

    return normalize_tablepredictor_cfg(model_cfg)
