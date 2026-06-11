import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ApartmentDataset(Dataset):
    def __init__(self,
                 dataset_type,
                 path,
                 data_transformer,
                 task,
                 include_target=False,
                 ) -> None:
        df = pd.read_csv(os.path.join(path, f"{dataset_type}.csv"))
        features = torch.as_tensor(
            data_transformer.transform(df)
        )

        first_feature_idx = int(not include_target)
        self.features = features[:, first_feature_idx:]

        self.num_samples = len(self.features)

        if task == 'train':
            self.target = torch.as_tensor(
                data_transformer.transform(df[['Стоимость']], target=True)
            )
            self.label = torch.as_tensor(df[['Стоимость']].values)
        elif task == 'pretrain':
            self.label = self.features
            num_bins = data_transformer.num_bins[first_feature_idx:]
            num_cats = data_transformer.num_cats
            offsets = np.cumsum(num_bins + num_cats)

            num_classes = sum(num_bins + num_cats)
            num_bins_len = len(num_bins)
            num_cats_len = len(num_cats)

            self.target = torch.cat(
                [
                    self.get_target(self.features[:, i], num_classes, num_bins[i], offsets[i: i + 2])
                    for i in range(num_bins_len)
                ] + [
                    self.get_target(self.features[:, i], num_classes)
                    for i in range(num_bins_len, num_bins_len + num_cats_len)
                ],
                dim=1
            )
        else:
            raise NotImplementedError()

        self.target = self.target.float()

    def get_target(self, labels, num_classes, smooth_range=None, id_range=None):
        if smooth_range is None:
            target = F.one_hot(labels, num_classes=num_classes)
        else:
            smooth_range = smooth_range // 2
            smooth_range = smooth_range + smooth_range % 2 + 1
            if smooth_range < 5:
                target = F.one_hot(labels, num_classes=num_classes)
            else:
                values = F.softmax(
                    torch.signal.windows.gaussian(smooth_range) * 8,
                    dim=0
                )
                target = torch.zeros(self.num_samples, num_classes)
                for i in range(self.num_samples):
                    label = labels[i].item()
                    l = max(label - smooth_range // 2, id_range[0])
                    r = min(label + smooth_range // 2 + 1, id_range[1])
                    target[i, l: r] = values[
                        max(0, smooth_range // 2 - label + l):
                        min(smooth_range, smooth_range // 2 - label + r)
                    ]
        return target.unsqueeze(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.target[idx],
            'label': self.label[idx]
        }


if __name__ == '__main__':
    from configs.pretrain_cfg import cfg
    from executors.trainer import Trainer

    trainer = Trainer(cfg)
    print(trainer.train_data[0]['mask'])
    print(trainer.train_data[0]['mask'])
    print(trainer.train_data[0]['mask'])
