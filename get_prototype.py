# coding=utf-8
"""
This script uses a checkpoint to extract the features of the training samples and calculate the prototype of each category, and finally saves these prototypes for subsequent prototype drift estimations. Please set up the data file (the data that will be extracted for features) in cub/cars/deepfashion2.py before running it.

An example:

CUDA_VISIBLE_DEVICES=0 python get_prototype.py  --resume ckps/cub/bn_inception-dim-512-lr1e-5-batchsize-80-test/ckp_ep2150.pth.tar --net bn_inception --data cub --output_file 1_100_proxy_s1

CUDA_VISIBLE_DEVICES=1 python get_prototype.py  --resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s1/ckp_ep360.pth.tar --net bn_inception --data deep2 --output_file 4239_8476_proxy_s1
"""

from __future__ import absolute_import, print_function

import argparse
import pickle

import numpy as np
import torch.utils.data
from torch.autograd import Variable
from torch.backends import cudnn
from tqdm import tqdm

import datasets
import models
from utils import load_checkpoint

cudnn.benchmark = True


def main(args):

    dim = args.dim
    model = models.create(args.net, pretrained=True, dim=dim)

    chk_pt = load_checkpoint(args.resume)
    weight = chk_pt['state_dict']
    model.load_state_dict(weight)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    data = datasets.create(args.data, ratio=args.ratio, width=args.width,
                           origin_width=args.origin_width, root=args.data_root, net=args.net)
    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size, pin_memory=True, num_workers=args.nThreads)

    embeddings = np.zeros((1, dim))
    embeddings_nonorm = np.zeros((1, dim))
    embeddings_labels = np.zeros((1))
    data_count = 0
    for inputs, labels in tqdm(train_loader):
        inputs = Variable(inputs).cuda()
        data_count += inputs.shape[0]
        with torch.no_grad():
            embed_feat, embed_feat_nonorm = model(inputs)
            embeddings_labels = np.concatenate(
                (embeddings_labels, labels.numpy()))
            embeddings = np.concatenate(
                (embeddings, embed_feat.cpu().numpy()), axis=0)
            embeddings_nonorm = np.concatenate(
                (embeddings_nonorm, embed_feat_nonorm.cpu().numpy()), axis=0)

    embeddings = embeddings[1:, :]
    embeddings_nonorm = embeddings_nonorm[1:, :]
    embeddings_labels = embeddings_labels[1:]
    assert embeddings.shape[0] == data_count
    assert embeddings_nonorm.shape[0] == data_count
    assert embeddings_labels.shape[0] == data_count

    labels_set = np.unique(embeddings_labels)
    class_mean = []
    class_mean_nonorm = []
    class_std = []
    class_label = []
    for i in labels_set:
        ind_cl = np.where(i == embeddings_labels)[0]
        embeddings_tmp = embeddings[ind_cl]
        embeddings_tmp_nonorm = embeddings_nonorm[ind_cl]
        class_label.append(i)
        class_mean.append(np.mean(embeddings_tmp, axis=0))
        class_mean_nonorm.append(np.mean(embeddings_tmp_nonorm, axis=0))
        class_std.append(np.std(embeddings_tmp_nonorm, axis=0))
    print("iterate {} samples!".format(data_count))
    print("get {} prototypes!".format(len(class_mean)))
    prototype = {'class_mean': class_mean, 'class_mean_nonorm': class_mean_nonorm,
                 'class_std': class_std, 'class_label': class_label}

    with open(args.output_file, 'wb') as f:
        pickle.dump(prototype, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimating Prototype Drift")

    parser.add_argument("--resume", type=str, default=None, required=True)
    parser.add_argument("--net", type=str,
                        default="bn_inception", help="backbone")
    parser.add_argument("--dim", type=int, default=512,
                        help="dimension of embedding")
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--data", type=str, default="cub", help="dataset name")
    parser.add_argument("--data_root", type=str,
                        default=None, help="path to dataset")
    parser.add_argument("--width", type=int, default=224,
                        help="width of input image")
    parser.add_argument("--origin_width", type=int,
                        default=256, help="size of original image")
    parser.add_argument("--ratio", type=float, default=0.16,
                        help="random crop ratio for training data")
    parser.add_argument("--nThreads", type=int, default=16,
                        help="number of data  loading threads")
    parser.add_argument("--output_file", type=str, default=None,
                        help="save the prototype drifts", required=True)

    print(parser.parse_args())
    main(parser.parse_args())
