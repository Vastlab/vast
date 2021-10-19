import torch
from ..tools import pairwisedistances


class EVM_activation:
    """
    Only implements the activation not the loss
    It is simply a re-write of the EVM_Inference function in opensetAlgos/EVM.py
    """

    def __init__(self, models, distance_metric, gpu):
        self.models = models
        self.distance_metric = distance_metric
        self.gpu = gpu

    def __call__(self, features):
        probs = torch.zeros((features.shape[0], len(self.models.keys()))).to(
            f"cuda:{self.gpu}"
        )
        for cls_no, cls_name in enumerate(sorted(self.models.keys())):
            distances = pairwisedistances.__dict__[self.distance_metric](
                features, self.models[cls_name]["extreme_vectors"].to(f"cuda:{self.gpu}")
            )
            probs_current_class = self.models[cls_name]["weibulls"].wscore(distances)
            probs[:, cls_no] = torch.max(probs_current_class, dim=1).values
        return probs
