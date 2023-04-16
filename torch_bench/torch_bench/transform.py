import torch
import numpy as np
from abc import ABC


class Transformation(ABC):
    def __init__(self):
        pass

    def __call__(self):
        pass


class to_tensor(Transformation):
    def __call__(self, image):
        if len(image.shape) == 3:
            return torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))
