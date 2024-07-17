import timm
import torch
import torch.nn as nn
from torch.autograd import Variable

from .inception import Embedding


class ViT(nn.Module):

    def __init__(self, dim=512, pretrained=True):
        super(ViT, self).__init__()
        self.dim = dim
        self.transformer = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        self.fc_layer = Embedding(384, self.dim, normalized=True)

    def forward(self, x):
        x =  self.transformer(x)
        fc_x, x_nonorm = self.fc_layer(x)
        return fc_x, x_nonorm


def vit(dim=512, pretrained=True):
    model = ViT(dim, pretrained)
    return model


def main():
    model = ViT()
    images = Variable(torch.randn(10, 3, 224, 224))
    out, _ = model(images)
    print(out.shape)

if __name__ == '__main__':
    main()