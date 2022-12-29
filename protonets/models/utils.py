import torch
import torch.nn as nn

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x, y):
    # x: N x D
    # y: M x D
    epsilon = 1e-8
    normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + epsilon)
    normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + epsilon)

    expanded_x = normalised_x.unsqueeze(1).expand(x.size(0), y.size(0), -1)
    expanded_y = normalised_y.unsqueeze(0).expand(x.size(0), y.size(0), -1)

    cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
    return 1 - cosine_similarities

def l1_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.abs(x - y).sum(2)
