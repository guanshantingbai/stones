from __future__ import absolute_import

from PIL import Image
from torchvision.transforms import *


class CovertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img
