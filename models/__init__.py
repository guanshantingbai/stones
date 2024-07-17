from .inception import bn_inception
from .vision_transformer import vit
from .resnet import resnet50

__factory = {
    'bn_inception': bn_inception,
    'vit': vit,
    'resnet50': resnet50,
}

def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        the name of network
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)

    return __factory[name](*args, **kwargs)
