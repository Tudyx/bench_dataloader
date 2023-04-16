# from src.dataloaders.base import DataLoader
from torchvision import transforms


from .dataset import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
    PytorchDataset,
)

import torch.utils.data as torch_data


DATASET = PytorchDataset()


class PytorchLoader:
    def _get(self, mode, transform, **kwargs):
        dataset = DATASET.get_local(mode=mode, transforms=transform)

        sampler = None

        loader = torch_data.DataLoader(dataset, **kwargs, sampler=sampler)
        return loader

    def get_train_loader(self, **kwargs):
        t = get_train_transforms()
        return self._get("train", t, **kwargs)

    def get_val_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("val", t, **kwargs)

    def get_test_loader(self, **kwargs):
        t = get_eval_transforms()
        return self._get("test", t, **kwargs)
