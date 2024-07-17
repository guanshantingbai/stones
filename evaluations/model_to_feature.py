# coding=utf-8
from __future__ import absolute_import, print_function

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from evaluations import extract_features
import models
import datasets
from utils import load_checkpoint
import timm
cudnn.benchmark = True


def model2feature(checkpoint, net, dim, data, root=None, width=224, batch_size=100, pool_feature=False, nThreads=16):
    model = models.create(net, pretrained=False, dim=dim)

    model_dict = model.state_dict()
    weights = checkpoint["state_dict"]
    pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model).cuda()
    data_name = data
    data = datasets.create(data_name, width=width, root=root, net=net)
    
    data_loader = DataLoader(data.test, batch_size=batch_size, pin_memory=True, num_workers=nThreads)

    features, labels = extract_features(model, data_loader, pool_feature=pool_feature)
    gallery_features, gallery_labels = query_features, query_labels = features, labels

    return gallery_features, gallery_labels, query_features, query_labels
