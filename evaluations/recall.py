# coding : utf-8
from __future__ import absolute_import, division

import random

import numpy as np
from utils import to_numpy


def recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['deep2'] = [1, 2, 4, 8, 16, 32]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape # m and n are related with the size of test set, sim_mat is computed from the extracted features
    gallery_ids = np.asarray(gallery_ids) #gallery labels
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    num_valid = np.zeros(len(k_s)) # len(k_s)) = 6
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])#x[gallery_ids == query_ids[i]]:only when gallery_ids==query_ids[i] is true,value in x is got
        neg_num = np.sum(x > pos_max)# the number of negative samples
        neg_nums[i] = neg_num # if neg_num=0, indicates that all negtative samples are regarded as positive samples

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]
    return num_valid / float(m)
