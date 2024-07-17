# coding=utf-8
from __future__ import absolute_import, print_function

import pickle
import time

import torch
from torch.autograd import Variable
from torch.backends import cudnn
import ot

from utils.meters import AverageMeter

cudnn.benchmark = True


def online_train(epoch, model, criterion, optimizer, train_loader, args):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    end = time.time()

    pair_loss, inter_, dist_ap, dist_an = 0.0, 0.0, 0.0, 0.0

    freq = min(args.print_freq, len(train_loader))

    model[0].train()
    for i, (inputs, labels) in enumerate(train_loader, 0):

        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat, _ = model[0](inputs)
        embed_feat_frozen, _ = model[1](inputs)

        # the task is not incremental learning
        # set loss function1, labels are for new task dataset
        pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels)
        distance_matrix = torch.cdist(embed_feat, embed_feat_frozen, p=3)
        student = torch.ones(len(embed_feat)).cuda() / len(embed_feat) * 1.0
        teacher = torch.ones(len(embed_feat_frozen)).cuda() / len(embed_feat_frozen) * 1.0
        T = ot.sinkhorn(student, teacher, distance_matrix, 0.464)
        embed_feat_frozen = torch.mm(T.T, embed_feat)
        similarity_loss = criterion[1](embed_feat, embed_feat_frozen)

        loss_correlation = 5 * similarity_loss
        loss = 1 * (pair_loss) + loss_correlation

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        accuracy.update(inter_)
        pos_sims.update(dist_ap)
        neg_sims.update(dist_an)

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Acc {accuracy.avg:.4f}\t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f}\t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, accuracy=accuracy, pos=pos_sims, neg=neg_sims))


def initial_train(epoch, model, criterion, optimizer, train_loader, args):

    losses = AverageMeter()
    batch_time = AverageMeter()
    accuracy = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    end = time.time()

    pair_loss, inter_, dist_ap, dist_an = 0.0, 0.0, 0.0, 0.0
    freq = min(args.print_freq, len(train_loader))

    model[0].train()
    for i, (inputs, labels) in enumerate(train_loader, 0):

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat, _ = model[0](inputs)
        pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels)
        loss = pair_loss

        if loss == 0.0:
            continue

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item())
        accuracy.update(inter_)
        pos_sims.update(dist_ap)
        neg_sims.update(dist_an)

        if (i + 1) % freq == 0 or (i+1) == len(train_loader):
            print('Epoch: [{0:03d}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f} \t'
                  'Acc {accuracy.avg:.4f}\t'
                  'Pos {pos.avg:.4f}\t'
                  'Neg {neg.avg:.4f}\t'.format
                  (epoch + 1, i + 1, len(train_loader), batch_time=batch_time,
                   loss=losses, accuracy=accuracy, pos=pos_sims, neg=neg_sims))

    return accuracy.avg
