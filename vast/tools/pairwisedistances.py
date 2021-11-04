import torch
import torch.nn as nn


def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum("nc,ck->nk", [x, y.T])
    distances = 1 - similarity
    return distances


# For euclidean distance compute_mode='use_mm_for_euclid_dist' or 'use_mm_for_euclid_dist_if_necessary'
# may provide a speedup but then the distance to the sample itself might not be zero.
# This is due to the precision error and the different equations solved in them vs the 'donot_use_mm_for_euclid_dist'
# When using mm approach the equation solved is x^2+y^2-2xy vs (x-y)^2
# This may cause precision errors and infact can also result in negative distances, though that is internally handled by pytorch.
# Source: https://github.com/pytorch/pytorch/issues/42479#issuecomment-668896488
def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    return distances

def l1(x, y):
    distances = torch.cdist(x, y, p=1.0, compute_mode="donot_use_mm_for_euclid_dist")
    return distances


__dict__ = {"cosine": cosine, "euclidean": euclidean, "l1", l1}
