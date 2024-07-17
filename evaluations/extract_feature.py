import torch
from torch.autograd import Variable
from utils import to_torch


def extract_batch_feature(model, inputs, pool_feature=False):
    model.eval()
    with torch.no_grad():
        inputs = to_torch(inputs)
        inputs = Variable(inputs).cuda()
        outputs, _ = model(inputs)
        return outputs


def extract_features(model, data_loader, pool_feature=False):
    feature_gpu = torch.FloatTensor().cuda()
    feature_cpu = torch.FloatTensor()
    
    trans_inter = 1e4
    labels = list()
    
    for i, (img, label) in enumerate(data_loader):
        outputs = extract_batch_feature(model, img, pool_feature=pool_feature)
        feature_gpu = torch.cat((feature_gpu, outputs.data), 0)
        labels.extend(label)
        count = feature_gpu.size(0)
        if count > trans_inter or i == len(data_loader) - 1:
            feature_cpu = torch.cat((feature_cpu, feature_gpu.cpu()), 0)
            feature_gpu = torch.FloatTensor().cuda()
            print("Extract Features: [{}/{}]".format(i + 1, len(data_loader)))
        del outputs
    return feature_cpu, labels

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x

def pairwise_similarity(x, y):
    y = normalize(y)
    x = normalize(x)
    similarity = torch.mm(x, y.t()) 
    return similarity