import os
import dill
import numpy as np
import pandas as pd


class DataTransformer:
    def __init__(self,
                 num_cfg,
                 cat_cfg,
                 target_cfg,
                 ) -> None:
        self.json = {
            'num_cfg': {k: v if isinstance(v, list) else str(v) for k, v in num_cfg.items()},
            'cat_cfg': {k: v if isinstance(v, list) else str(v) for k, v in cat_cfg.items()},
            'target_cfg': {k: v if isinstance(v, list) else str(v) for k, v in target_cfg.items()},
        }
        self.num_path = num_cfg['path']
        self.cat_path = cat_cfg['path']
        self.target_path = target_cfg['path']

        self.num_processor = num_cfg['processor']
        self.cat_processor = cat_cfg['processor']
        self.target_processor = target_cfg['processor']

        self.num_cols = num_cfg['columns']
        self.cat_cols = cat_cfg['columns']
        self.target_cols = target_cfg['columns']

        if os.path.exists(num_cfg['path']):
            with open(num_cfg['path'], 'rb') as f:
                self.num_processor = dill.load(f)
        if os.path.exists(cat_cfg['path']):
            with open(cat_cfg['path'], 'rb') as f:
                self.cat_processor = dill.load(f)
        if os.path.exists(target_cfg['path']):
            with open(target_cfg['path'], 'rb') as f:
                self.target_processor = dill.load(f)
        self._set_params()

    def save(self):
        with open(self.num_path, 'wb') as f:
            dill.dump(self.num_processor, f)
        with open(self.cat_path, 'wb') as f:
            dill.dump(self.cat_processor, f)
        with open(self.target_path, 'wb') as f:
            dill.dump(self.target_processor, f)

    def _set_params(self):
        cats = [
            len(c) for c in self.cat_processor.categories_
        ]
        inf_cats = [
            0 if c is None else len(c) - 1
            for c in self.cat_processor.infrequent_categories_
        ]
        self.num_cats = [i - j for i, j in zip(cats, inf_cats)]
        self.num_bins = self.num_processor.n_bins_.tolist()
        # num_classes = self.num_bins + self.num_cats
        # self.offsets = np.cumsum([0] + num_classes[:-1])

    def fit(self, data):
        self.num_processor.fit(data[self.num_cols].fillna(-1.0))
        self.cat_processor.fit(data[self.cat_cols])
        self.target_processor.fit(data[self.target_cols])
        self._set_params()

    def transform(self, data, target=False):
        if target:
            return self.target_processor.transform(data)
        else:
            num = self.num_processor.transform(data[self.num_cols].fillna(-1.0))
            # if not self.include_target_in_features:
            #     num = num[:, 1:]
            cat = self.cat_processor.transform(data[self.cat_cols])
            for i, c in enumerate(self.num_cats):
                cat[cat[:, i] == -1, i] = c - 1
            data = np.hstack([num, cat])
            # if self.apply_offsets:
            #     data = data + self.offsets
            return data.astype(int)

    def inverse_transform(self, data, target=None, numpy=True):
        if target is None:
            if data.ndim < 2:
                data = np.reshape(data, [1, -1])

            i = len(self.num_cols)
            if data.shape[1] < len(self.num_cols + self.cat_cols):
                data = np.hstack([np.zeros([data.shape[0], 1]), data])

            num = self.num_processor.inverse_transform(data[:, :i])
            num[num < 0] = np.nan
            cat = self.cat_processor.inverse_transform(data[:, i:])
            data = np.hstack([num, cat])
            if numpy:
                return data
            else:
                return pd.DataFrame(data, columns=self.num_cols + self.cat_cols)
        else:
            if target == 'num':
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data, columns=self.target_cols)
                data = self.target_processor.inverse_transform(data)
            elif target == 'cat':
                data = np.hstack([data.argmax(-1).reshape(-1, 1),
                                  np.zeros([len(data), len(self.num_cols) - 1])])
                if not isinstance(data, pd.DataFrame):
                    data = pd.DataFrame(data, columns=self.num_cols)
                data = self.num_processor.inverse_transform(data)
                data = np.expand_dims(data[:, 0], axis=1)
            else:
                raise NotImplementedError()
            if numpy:
                return data
            else:
                return pd.DataFrame(data, columns=self.target_cols)

    # def __json__(self):
    #     return self._json



columns = [
    'Тип продажи',
    'Объект продажи',
    'Общая площадь',
    'Жилая площадь',
    'Площадь кухни',
    'Этаж',
    'Этажей в доме',
    'Лифт пассажирский (кол-во)',
    'Лифт грузовой (кол-во)',
    'Год строительства',
    'Балкон/лоджия',
    'Мусоропровод',
    'Парковка',
    'Количество комнат',
    'Тип дома',
    'Высота потолков',
    'Кол-во раздельных санузлов',
    'Кол-во совмещенных санузлов',
    'Ремонт',
    'Вид из окон',
    'Расстояние до метро',
    'Кол-во квартир в доме',
    'Адрес',
    'Стоимость',
    'Регион Циан'
]


if __name__ == '__main__':
    import pandas as pd
    from configs.data_cfg import cfg
    dt = cfg.data_transformer

    for col, edges in zip(dt.num_cols, dt.num_processor.bin_edges_):
        print(col, edges)
