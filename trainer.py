# coding=utf-8
from __future__ import absolute_import, print_function

import pickle
import time

import torch
from torch.autograd import Variable
from torch.backends import cudnn

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

    if Feature_Estimation_flag:
        with open('1_4238_esti_proxy_s3', 'rb') as f:
            prototype_s1 = pickle.load(f)
        class_mean_s1 = Variable(torch.FloatTensor(
            prototype_s1['class_mean'])).cuda()
        class_drift_s1 = Variable(torch.FloatTensor(
            prototype_s1['class_drift'])).cuda()

        with open('4239_8476_esti_proxy_s3', 'rb') as f:
            prototype_s2 = pickle.load(f)
        class_mean_s2 = Variable(torch.FloatTensor(
            prototype_s2['class_mean'])).cuda()
        class_drift_s2 = Variable(torch.FloatTensor(
            prototype_s2['class_drift'])).cuda()

        with open('8477_12714_esti_proxy_s3', 'rb') as f:
            prototype_s3 = pickle.load(f)
        class_mean_s3 = Variable(torch.FloatTensor(
            prototype_s3['class_mean'])).cuda()
        class_drift_s3 = Variable(torch.FloatTensor(
            prototype_s3['class_drift'])).cuda()

    model[0].train()
    for i, (inputs, labels) in enumerate(train_loader, 0):

        # wrap them in Variable
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()

        embed_feat, _ = model[0](inputs)
        embed_feat_frozen, embed_feat_frozen_nonorm = model[1](inputs)
        embed_feat_new, _ = model[2](inputs)

        # the task is not incremental learning
        # set loss function1, labels are for new task dataset
        pair_loss, inter_, dist_ap, dist_an = criterion[0](embed_feat, labels)
        pair_loss_new, inter_new, dist_ap_new, dist_an_new = criterion[0](
            embed_feat_new, labels)

        if Feature_Estimation_flag:
            # estimate 126-150 features
            weight3 = torch.matmul(embed_feat_frozen, class_mean_s3.T)
            weight_norm3 = torch.unsqueeze(torch.sum(weight3, dim=1), -1)
            feature_drift3 = torch.matmul(
                weight3, class_drift_s3) / weight_norm3
            estimation_feat3_nonorm = embed_feat_frozen_nonorm - feature_drift3
            assert estimation_feat3_nonorm.shape == embed_feat_frozen.shape
            feat3_norm = estimation_feat3_nonorm.norm(dim=1, p=2, keepdim=True)
            estimation_feat3 = estimation_feat3_nonorm.div(
                feat3_norm.expand_as(estimation_feat3_nonorm))
            correlation_loss3 = criterion[1](embed_feat, estimation_feat3) * 10

            # # estimate 101-125 features
            weight2 = torch.matmul(embed_feat_frozen, class_mean_s2.T)
            weight_norm2 = torch.unsqueeze(torch.sum(weight2, dim=1), -1)
            feature_drift2 = torch.matmul(
                weight2, class_drift_s2) / weight_norm2
            estimation_feat2_nonorm = embed_feat_frozen_nonorm - feature_drift2
            assert estimation_feat2_nonorm.shape == embed_feat_frozen.shape
            feat2_norm = estimation_feat2_nonorm.norm(dim=1, p=2, keepdim=True)
            estimation_feat2 = estimation_feat2_nonorm.div(
                feat2_norm.expand_as(estimation_feat2_nonorm))
            correlation_loss2 = criterion[1](embed_feat, estimation_feat2) * 10

            # estimate 1-100 features
            weight = torch.matmul(embed_feat_frozen, class_mean_s1.T)
            weight_norm = torch.unsqueeze(torch.sum(weight, dim=1), -1)
            feature_drift = torch.matmul(weight, class_drift_s1) / weight_norm
            estimation_feat1 = embed_feat_frozen_nonorm - feature_drift
            assert estimation_feat1.shape == embed_feat_frozen.shape
            feat1_norm = estimation_feat1.norm(dim=1, p=2, keepdim=True)
            estimation_feat1 = estimation_feat1.div(
                feat1_norm.expand_as(estimation_feat1))
            correlation_loss1 = criterion[1](embed_feat, estimation_feat1) * 10

        similarity_loss = criterion[1](embed_feat, embed_feat_frozen)
        similarity_loss2 = criterion[1](embed_feat_new, embed_feat)
        similarity_loss3 = criterion[1](embed_feat, embed_feat_new)

        loss_correlation = 10 * similarity_loss + (similarity_loss2 + similarity_loss3) * 0.5 * args.mutual \
            + correlation_loss1 + correlation_loss2 + correlation_loss3
        loss = 1 * (pair_loss) + loss_correlation + 1 * (pair_loss_new)

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
