import torch
import torch.nn as nn

def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum('nc,ck->nk', [x, y.T])
    distances = 1-similarity
    return distances

def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    return distances

__dict__ = {'cosine':cosine,
            'euclidean':euclidean
           }
