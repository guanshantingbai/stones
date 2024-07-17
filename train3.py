'''
Thanks for the code release from WangXun from: https://github.com/bnu-wangxun/Deep_Metric
if use this code, please consider the papers:

@article{chen2021feature,
title={Feature Estimations based Correlation Distillation for Incremental Image Retrieval},
author={Wei Chen and Yu Liu and Nan Pu and Weiping Wang and Li Liu and Lew Michael S},
journal={IEEE Transactions on Multimedia},
year={2021},
}

@inproceedings{wang2019multi,
title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
author={Wang Xun and Han Xintong and Huang Weilin and Dong Dengke and Scott Matthew R},
booktitle={CVPR},
year={2019}
}
}

'''

# coding=utf-8
from __future__ import absolute_import, print_function

import argparse
import os
import os.path as osp
import sys

import torch
import torch.utils.data
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader

import datasets
import losses
import models
from evaluations import validate
from losses import correlation_loss
from trainer3 import initial_train, online_train
from utils import Logger, FastRandomIdentitySampler, display, mkdir_if_missing, save_checkpoint, set_bn_eval

cudnn.benchmark = True

use_gpu = True


def main(args):
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)
    log_name = "log.txt"
    sys.stdout = Logger(os.path.join(save_dir, log_name))
    display(args)
    start_epoch = 0

    # prepare dataset
    data = datasets.create(args.data, args.data_root, width=args.width,
                           origin_width=args.origin_width, ratio=args.ratio, net=args.net)
    train_loader = DataLoader(data.train, batch_size=args.batch_size,
                              sampler=FastRandomIdentitySampler(
                                  data.train, num_instances=args.num_instances),
                              drop_last=True, pin_memory=True, num_workers=args.nThreads)
    test_loader = DataLoader(data.test, batch_size=args.batch_size,
                             pin_memory=True, num_workers=args.nThreads)
    origin_test_loader = DataLoader(data.origin_test, batch_size=args.batch_size,
                                    pin_memory=True, num_workers=args.nThreads)

    # build model
    model = models.create(args.net, dim=args.dim, pretrained=True)
    model_teacher = models.create(args.net, dim=args.dim, pretrained=True)

    # resume checkpoint or not
    if args.resume is not None:
        print("load checkpoint from {}".format(args.resume))

        model_dict = model.state_dict()
        model_teacher_dict = model_teacher.state_dict()
        checkpoint = torch.load(args.resume)
        weight = checkpoint["state_dict"]
        start_epoch = checkpoint["epoch"]

        pretrained_dict = {k: v for k, v in weight.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        pretrained_dict_teacher = {
            k: v for k, v in weight.items() if k in model_teacher_dict}
        model_teacher_dict.update(pretrained_dict_teacher)
        model_teacher.load_state_dict(model_teacher_dict)
        model_teacher.eval()

    model = torch.nn.DataParallel(model).cuda()
    model_teacher = torch.nn.DataParallel(model_teacher).cuda()

    # frozen the model_teacher
    frozen_params = [p for p in model_teacher.module.parameters()]
    for p in frozen_params:
        p.requires_grad = False

    # fine-tuning the model and model_supporting
    # freeze BN
    if args.freeze_BN is True:
        model.apply(set_bn_eval)

    new_param_ids_fc_layer = set(
        map(id, model.module.fc_layer.parameters()))

    new_param_ids = new_param_ids_fc_layer

    new_params_fc = [p for p in model.module.parameters() if id(
        p) in new_param_ids_fc_layer]
    base_params = [p for p in model.module.parameters() if id(p)
                   not in new_param_ids]

    param_groups = [
        {'params': base_params, 'lr_mult': 0.1},
        {'params': new_params_fc, 'lr_mult': 1.0},]

    # build optimizer and losses
    optimizer = Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    training_loss = losses.create(args.loss).cuda()
    correlation = correlation_loss().cuda()
    loss_list = [training_loss, correlation]
    model_list = [model, model_teacher]

    if args.online_training:
        train = online_train
    else:
        train = initial_train

    # training
    for epoch in range(start_epoch, args.epochs):
        train(epoch=epoch, model=model_list, criterion=loss_list,
              optimizer=optimizer, train_loader=train_loader, args=args)

        if (epoch + 1) % args.save_step == 0 or epoch == 0:
            # validating
            print('\n')
            validate(model, args.data, test_loader, epoch, False)
            validate(model, args.data, origin_test_loader, epoch, False)
            print('\n')

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({"state_dict": state_dict, "epoch": (epoch + 1)},
                            fpath=osp.join(args.save_dir, "ckp_ep" + str(epoch + 1) + ".pth.tar"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online Deep Metric Learning via Mutual Distillation Training")

    # hype-parameters
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=80,
                        help="mini-batch size", metavar="n")
    parser.add_argument("--num_instances", type=int, default=5,
                        help="number of samples from one class in a mini-batch")
    parser.add_argument("--epochs", type=int, default=1500,
                        help="epochs for training")
    parser.add_argument("--dim", type=int, default=512,
                        help="dimension of embedding", metavar='n')
    parser.add_argument("--mutual", type=int, default=1,
                        help="mutual param for mutual distillation loss")

    # data
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
                        help="number of data loading threads")

    # model
    parser.add_argument("--net", type=str,
                        default="bn_inception", help="backbone")
    parser.add_argument("--freeze_BN", type=bool,
                        default=True, help="Freeze BN or not")

    # online training
    parser.add_argument("--online_training", type=bool,
                        default=False, help="online training or not")
    parser.add_argument("--feature_estimation", type=bool,
                        default=False, help="estimating features or not")
    parser.add_argument(
        "--loss", type=str, default="hard_triplet", help="loss for training network")
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    # checkpoint
    parser.add_argument("--save_step", type=int, default=50,
                        help="number of epochs to save the model")
    parser.add_argument("--resume", type=str, default=None,
                        help="resume checkpoint")

    # others
    parser.add_argument("--print_freq", type=int, default=6,
                        help="display frequency during training")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="dir to save checkpoints and log")

    main(parser.parse_args())
