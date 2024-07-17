import torch

from evaluations import extract_features, pairwise_similarity, recall_at_ks


def validate(model, data, test_loader, epoch, pool_feature=False):
    features, labels = extract_features(model, test_loader, pool_feature)
    sim_mat = pairwise_similarity(features, features)
    sim_mat = sim_mat - torch.eye(sim_mat.size(0))
    recall_ks = recall_at_ks(sim_mat, data, labels, labels)
    result = "  ".join(["%.4f" % k for k in recall_ks])
    print("Epoch-%d" % (epoch + 1), result)