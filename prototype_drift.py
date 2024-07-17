# coding=utf-8
"""
This script calculates the feature drift of data between the two models (old_resume and new_resume), and works with the original prototypes of the data on the old model to estimate the drift of these prototypes, and finally updates the original prototypes. Please set up the data file (the data that will be extracted for features) in cub/cars/deepfashion2.py before running it.

An example:

CUDA_VISIBLE_DEVICES=0 python prototype_drift.py  --old_resume ckps/cub/bn_inception-dim-512-lr1e-5-batchsize-80-multi-task-s1/best_model.pth.tar --new_resume ckps/cub/bn_inception-dim-512-lr1e-5-batchsize-80-multi-task-s3/best_model.pth.tar --prototype 101_125_proxy_s1 --net bn_inception --data cub --output_file 101_125_esti_proxy_s3

CUDA_VISIBLE_DEVICES=1 python prototype_drift.py  --old_resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-origin/ckp_ep170.pth.tar --new_resume ckps/deep2/bn_inception-hardmining-lr1e-5-batchsize-80-s1/ckp_ep360.pth.tar --prototype proxy/deep2/1_4238_proxy --net bn_inception --data deep2 --output_file 1_4238_esti_proxy_s1
"""
from __future__ import absolute_import, print_function
import torch.utils.data
from torch.backends import cudnn
from torch.autograd import Variable
import models
from utils import load_checkpoint

import datasets
import numpy as np
import pickle
import argparse
from tqdm import tqdm
cudnn.benchmark = True


def main(args):

    dim = args.dim

    old_model = models.create(args.net, pretrained=False, dim=dim)
    new_model = models.create(args.net, pretrained=False, dim=dim)

    old_chk_pt = load_checkpoint(args.old_resume)
    old_weight = old_chk_pt['state_dict']
    old_model.load_state_dict(old_weight)

    new_chk_pt = load_checkpoint(args.new_resume)
    new_weight = new_chk_pt['state_dict']
    new_model.load_state_dict(new_weight)

    old_model = torch.nn.DataParallel(old_model).cuda()
    old_model.eval()

    new_model = torch.nn.DataParallel(new_model).cuda()
    new_model.eval()
    
    with open(args.prototype, 'rb') as f:
        prototype = pickle.load(f)

    data = datasets.create(args.data, root=args.data_root, ratio=args.ratio,
                           width=args.width, origin_width=args.origin_width, net=args.net)

    train_loader = torch.utils.data.DataLoader(
        data.train, batch_size=args.batch_size, pin_memory=True, num_workers=args.nThreads)

    old_embeddings = np.zeros((1, dim))
    new_embeddings = np.zeros((1, dim))
    old_embeddings_nonorm = np.zeros((1, dim))
    new_embeddings_nonorm = np.zeros((1, dim))
    for inputs, labels in tqdm(train_loader):
        inputs = Variable(inputs).cuda()

        with torch.no_grad():
            old_embed_feat, old_embed_feat_nonorm = old_model(inputs)
            new_embed_feat, new_embed_feat_nonorm = new_model(inputs)

            old_embeddings = np.concatenate(
                (old_embeddings, old_embed_feat.cpu().numpy()), axis=0)
            old_embeddings_nonorm = np.concatenate(
                (old_embeddings_nonorm, old_embed_feat_nonorm.cpu().numpy()), axis=0)
            new_embeddings = np.concatenate(
                (new_embeddings, new_embed_feat.cpu().numpy()), axis=0)
            new_embeddings_nonorm = np.concatenate(
                (new_embeddings_nonorm, new_embed_feat_nonorm.cpu().numpy()), axis=0)

    old_embeddings = old_embeddings[1:, :]
    old_embeddings_nonorm = old_embeddings_nonorm[1:, :]
    new_embeddings = new_embeddings[1:, :]
    new_embeddings_nonorm = new_embeddings_nonorm[1:, :]

    assert old_embeddings.shape == new_embeddings.shape
    assert old_embeddings_nonorm.shape == old_embeddings.shape
    drifts = new_embeddings_nonorm - old_embeddings_nonorm  

    class_mean = np.array(prototype['class_mean'], dtype=float)
    class_mean_nonorm = np.array(prototype['class_mean_nonorm'], dtype=float)
    class_std = prototype['class_std']
    class_labels = prototype['class_label']
    assert len(class_mean) == len(class_std)
    assert len(class_mean_nonorm) == len(class_mean)

    assert class_mean.shape[1] == old_embeddings.shape[1]
    weight = np.matmul(class_mean, old_embeddings.T)
    weight_norm = np.expand_dims(np.sum(weight, axis=1), axis=1)
    class_drifts = np.matmul(weight, drifts) / weight_norm
    new_class_mean_nonorm = class_mean_nonorm + class_drifts
    new_class_mean = new_class_mean_nonorm / \
        np.linalg.norm(new_class_mean_nonorm, 2, axis=1, keepdims=True)

    new_prototype = {'class_mean': new_class_mean, 'class_mean_nonorm': new_class_mean_nonorm,
                     'class_std': class_std, 'class_label': class_labels, 'class_drift': class_drifts}
    with open(args.output_file, 'wb') as f:
        pickle.dump(new_prototype, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Estimating Prototype Drift")

    parser.add_argument("--old_resume", type=str, default=None,
                        help="the model of an old task", required=True)
    parser.add_argument("--new_resume", type=str, default=None,
                        help="the model of a new task", required=True)
    parser.add_argument("--prototype", type=str, default=None,
                        help="the prototype file of the old task")
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
