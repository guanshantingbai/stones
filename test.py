# coding=utf-8
from __future__ import absolute_import, print_function
import argparse

from evaluations import pairwise_similarity, recall_at_ks, model2feature
from utils import load_checkpoint
import torch


def main(args):
    checkpoint = load_checkpoint(args.resume)
    epoch = checkpoint["epoch"]

    gallery_features, gallery_labels, query_features, query_labels = model2feature(
        checkpoint, args.net, args.dim, args.data, args.data_root, args.width, args.batch_size, args.pool_feature, args.nThreads)

    sim_mat = pairwise_similarity(query_features, gallery_features)

    if args.gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks = recall_at_ks(sim_mat, args.data, query_labels, gallery_labels)
    result = '  '.join(["%.4f" % k for k in recall_ks])
    print("Epoch-%d" % epoch, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online Deep Metric Learning via Mutual Distillation Testing")

    parser.add_argument("--resume", type=str, default=None,
                        help="testing checkpoint")
    parser.add_argument("--net", type=str,
                        default="bn_inception", help="backbone")
    parser.add_argument("--dim", type=int, default=512,
                        help="dimension of embedding")
    parser.add_argument("--data", type=str, default="cub", help="dataset name")
    parser.add_argument("--data_root", type=str,
                        default=None, help="path to dataset")
    parser.add_argument("--width", type=int, default=224,
                        help="width of input image")
    parser.add_argument("--gallery_eq_query", type=bool,
                        default=True, help="Is gallery identical with query")
    parser.add_argument("--batch_size", type=int, default=80,
                        help="mini-batch size", metavar="n")
    parser.add_argument("--pool_feature", type=bool, default=False)
    parser.add_argument("--nThreads", type=int, default=16,
                        help="number of data loading threads")

    # print("parser: ", parser.parse_args())
    main(parser.parse_args())
