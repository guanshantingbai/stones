import torch
import torch.nn as nn
import torch.nn.functional as F


class correlation_loss(nn.Module):
    """
    https://ieeexplore.ieee.org/abstract/document/9410400
    """
    def __init__(self, num_class=0):
        super(correlation_loss, self).__init__()
        self.use_gpu = True
        self.T = 4

    def forward(self, FeaT, FeaS):
        # calculate the similar matrix
        if self.use_gpu:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.cuda.FloatTensor) # label-free
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.cuda.FloatTensor)
        else:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.FloatTensor)
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.FloatTensor)

        # kl divergence
        p_s = F.log_softmax(Sim_S / self.T, dim=1)
        p_t = F.softmax(Sim_T / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]

        # reverse KL
        # p_s = F.log_softmax(Sim_T / self.T, dim=1)
        # p_t = F.softmax(Sim_S / self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]
        return loss

class absdistance_loss(nn.Module):
    """
    https://ieeexplore.ieee.org/abstract/document/9410400
    """
    def __init__(self, num_class=0):
        super(absdistance_loss, self).__init__()
        self.use_gpu = True
        self.T = 4

    def forward(self, FeaT, FeaS):

        # kl divergence
        p_s = F.log_softmax(FeaS / self.T, dim=1)
        p_t = F.softmax(FeaT / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / FeaS.shape[0]
        
        return loss

class symmetry_correlation_loss(nn.Module):
    def __init__(self, num_class=0):
        super(symmetry_correlation_loss, self).__init__()
        self.use_gpu = True
        self.T = 4

    def forward(self, FeaT, FeaS):
        # calculate the similar matrix
        if self.use_gpu:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.cuda.FloatTensor) # label-free
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.cuda.FloatTensor)
        else:
            Sim_T = torch.mm(FeaT, FeaT.t()).type(torch.FloatTensor)
            Sim_S = torch.mm(FeaS, FeaS.t()).type(torch.FloatTensor)

        p_s = F.log_softmax(Sim_S / self.T, dim=1)
        p_t = F.softmax(Sim_T / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]

        p_s = F.log_softmax(Sim_T / self.T, dim=1)
        p_t = F.softmax(Sim_S / self.T, dim=1)
        loss += F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]
        return loss / 2

class correlation_loss2(nn.Module):
    def __init__(self, num_class=0):
        super(correlation_loss2, self).__init__()
        self.use_gpu = True
        self.T = 4

    def forward(self, feat, label):
        # calculate the similar matrix
        feat_clone = feat.data.clone()
        if self.use_gpu:
            Sim_T = torch.mm(feat, feat.t()).type(torch.cuda.FloatTensor)
            sim_mat = torch.matmul(feat_clone, feat_clone.t()).type(torch.cuda.FloatTensor)
        else:
            Sim_T = torch.mm(feat, feat.t()).type(torch.FloatTensor)
            sim_mat = torch.matmul(feat_clone, feat_clone.t()).type(torch.FloatTensor)

        targets = label
        n = feat.size(0)
        
        for i in range(n):
            pos_value = torch.max(sim_mat[i]) - 0.1
            neg_value = torch.max(sim_mat[i]) + 0.1
            pos_mask = targets == targets[i]
            neg_mask = targets != targets[i]
            sim_mat[i][pos_mask] = pos_value
            sim_mat[i][neg_mask] = neg_value
            
        Sim_S = sim_mat

        # kl divergence
        p_s = F.log_softmax(Sim_S / self.T, dim=1)
        p_t = F.softmax(Sim_T / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / Sim_S.shape[0]

        return loss