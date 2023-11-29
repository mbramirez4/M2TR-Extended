import os

import torch

from M2TR.datasets.dataset import DeepFakeDataset
from M2TR.utils.registries import DATASET_REGISTRY

from .utils import get_image_from_path

'''
DATASET:
  DATASET_NAME: Custom
  ROOT_DIR: /some_where/data
  TRAIN_INFO_TXT: '/some_where/data/splits/train.txt'
  VAL_INFO_TXT: '/some_where/data/splits/eval.txt'
  TEST_INFO_TXT: '/some_where/data/splits/test.txt'
'''


@DATASET_REGISTRY.register()
class Custom(DeepFakeDataset):
    def __getitem__(self, idx):
        info_line = self.info_list[idx]
        image_path = info_line.strip()
        image_abs_path = os.path.join(self.root_dir, image_path)

        img, _ = get_image_from_path(
            image_abs_path, None, self.mode, self.dataset_cfg
        )
        img_label_binary = int("real" in image_path)

        sample = {
            'img': img,
            'bin_label': [int(img_label_binary)],
        }

        sample['img'] = torch.FloatTensor(sample['img'])
        sample['bin_label'] = torch.FloatTensor(sample['bin_label'])
        sample['bin_label_onehot'] = self.label_to_one_hot(
            sample['bin_label'], 2
        ).squeeze()
        sample['img_path'] = image_path
        return sample
