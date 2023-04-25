from .dataset.random import (
    get_train_transforms,
    get_eval_transforms,
    LABELS_DICT,
    RandomDataset,
)

from .dataset.random_unique import RandomUniqueDataset

import torch.utils.data as torch_data


class PytorchLoader:
    def __init__(self, dataset: str):
        if dataset == "random":
            self.dataset = RandomDataset()
        elif dataset == "random-unique":
            self.dataset = RandomUniqueDataset()
        else:
            raise ValueError(f"Unknown dataset type {dataset}")

    def _get(self, mode, transform, **kwargs):
        dataset = self.dataset.get_local(mode=mode, transforms=transform)

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
