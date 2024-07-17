from __future__ import absolute_import, print_function

from .binomial import binomial_loss
from .hard_triplet import hard_triplet_loss, hard_mining_loss
from .contrastive import contrastive_loss
from .correlation import correlation_loss, symmetry_correlation_loss, correlation_loss2, absdistance_loss

__factory = {
    'hard_triplet': hard_triplet_loss,
    'hard_mining': hard_mining_loss,
    'binomial': binomial_loss,
    'contrastive': contrastive_loss,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a loss instance.

    Parameters
    ----------
    name : str
        the name of loss function
    """
    if name not in __factory:
        raise KeyError("Unknown loss:", name)
    return __factory[name](*args, **kwargs)