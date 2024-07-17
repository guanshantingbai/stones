# coding=utf-8
from __future__ import absolute_import, print_function

import pickle
import time

import torch
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np

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
    Feature_Estimation_flag = args.feature_estimation

    model[0].train()
    # class_drifts_epoch = torch.zeros_like(class_mean_nonorm).cuda()
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat, embed_feat_nonorm = model[0](inputs)
        embed_feat_frozen, embed_feat_frozen_nonorm = model[1](inputs)
        # embed_feat_new, _ = model[2](inputs)
        similarity_loss = criterion[1](embed_feat, embed_feat_frozen)
        # similarity_loss2 = criterion[1](embed_feat_new, embed_feat)
        # similarity_loss3 = criterion[1](embed_feat, embed_feat_new)

        similarity_loss4 = 0.0
        sim = torch.matmul(embed_feat, embed_feat_frozen.T)
        for k in range(inputs.size(0)):
            similarity_loss4 += 1 - sim[k][k]
        similarity_loss4 /= inputs.size(0)
        
        # similarity_loss5 = 0.0
        # sim2 = torch.matmul(embed_feat, embed_feat_new.T)
        # for k in range(inputs.size(0)):
        #     similarity_loss5 += 1 - sim2[k][k]
        # similarity_loss5 /= inputs.size(0)
        
        pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels)
        # pair_loss_new, inter_new, dist_ap_new, dist_an_new = criterion[0](embed_feat_new, labels)
        loss_correlation = 0.2 * similarity_loss4 + 10 * similarity_loss
        # loss_correlation = 10 * similarity_loss
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
            
        # model[0].eval()
        # with torch.no_grad():  
        #     _, embed_feat_new_nonorm = model[0](inputs)
        #     drifts = embed_feat_new_nonorm - embed_feat_nonorm
        #     weight = torch.matmul(class_mean, embed_feat_.T)
        #     weight_norm = torch.unsqueeze(torch.sum(weight, dim=1), -1)
        #     class_drifts_batch = torch.matmul(weight, drifts) / weight_norm
            # class_drifts_epoch += class_drifts_batch 
            # class_drifts_epoch = class_drifts_epoch / len(train_loader)
            # new_class_mean_nonorm = class_mean_nonorm + class_drifts_epoch
            # new_class_mean_nonorm = 0.99 * class_mean_nonorm + 0.01 * class_drifts_batch
            # norm2 = new_class_mean_nonorm.norm(dim=1, p=2, keepdim=True)
            # new_class_mean = new_class_mean_nonorm.div(norm2.expand_as(new_class_mean_nonorm))
            # class_mean = new_class_mean
            # class_mean_nonorm = new_class_mean_nonorm
            
    # class_drifts_epoch = class_drifts_epoch / len(train_loader)
    # new_class_mean_nonorm = class_mean_nonorm + class_drifts_epoch
    # norm2 = new_class_mean_nonorm.norm(dim=1, p=2, keepdim=True)
    # new_class_mean = new_class_mean_nonorm.div(norm2.expand_as(new_class_mean_nonorm))
    # class_mean = new_class_mean
    # class_mean_nonorm = new_class_mean_nonorm

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
