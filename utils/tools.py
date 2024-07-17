from __future__ import absolute_import, print_function

import errno
import os
import os.path as osp

import torch


def mkdir_if_missing(dpath):
    try:
        os.mkdir(dpath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("No checkpoint found at '{}'".format(fpath))


def save_checkpoint(state, fpath="checkpoint.pth.tar"):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def display(args):
    #  Display information of current training
    print('Learning Rate  \t%.1e' % args.lr)
    print('Epochs  \t%05d' % args.epochs)
    print('Save Step  \t%03d' % args.save_step)
    print('Log Path \t%s' % args.save_dir)
    print('Network \t %s' % args.net)
    print('BatchNorm frozen  \t %s' % args.freeze_BN)
    print('Data Set \t %s' % args.data)
    print('Batch Size  \t %d' % args.batch_size)
    print('Num-Instance  \t %d' % args.num_instances)
    print('Embedded Dimension \t %d' % args.dim)
    print('Loss Function \t %s' % args.loss)
    print('Online Training  \t %s' % args.online_training)
    print('Feature Estimating \t %s' % args.feature_estimation)
    print('Begin to fine tune %s Network' % args.net)

    print(40*'#')
