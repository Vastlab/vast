# from .to_categorical import to_categorical
# from .iou import iou
from .ConcatDataset import ConcatDataset
from .features_dict_to_dim_dict import features_to_dim
# from .visualizing_tools import *
# from .evaluation_tools import *


_device = None


def device(x):
    global _device
    if _device is None:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return x.to(_device)


def set_device_cpu():
    global _device
    _device = "cpu"
