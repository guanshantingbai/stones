from .cub import cub200
from .cars import cars196
from .deepfashion2 import deepfashion2
from .custom_dataset import custom, generate_transform_dict
import os 

__factory = {
    "cub": cub200,
    "car": cars196,
    "deep2": deepfashion2,
}


def names():
    return sorted(__factory.keys())

def get_full_name(name):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name].__name__

def create(name, root=None, *args, **kwargs):
    """
    Create a dataset instance.
    """
    if root is not None:
        root = os.path.join(root, get_full_name(name))

    if name not in __factory:

        raise KeyError("Unknown dataset:", name)
    return __factory[name](root=root, *args, **kwargs)
