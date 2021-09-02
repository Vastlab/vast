import torch
from torch.nn import functional as F


def openmax_alpha(activations, evt_probs, alpha=1):
    """
    Algorithm 2 OpenMax probability estimation with rejection of
    unknown or uncertain inputs.
    Require: Activation vector for v(x) = v1(x), . . . , vN (x)
    Require: means µj and libMR models ρj = (τi, λi, κi)
    Require: α, the numer of “top” classes to revise
    1: Let s(i) = argsort(vj (x)); Let ωj = 1
    2: for i = 1, . . . , α do
    3:     ωs(i)(x) = 1 − ((α−i)/α)*e^−((||x−τs(i)||/λs(i))^κs(i))
    4: end for
    5: Revise activation vector vˆ(x) = v(x) ◦ ω(x)
    6: Define vˆ0(x) = \sum_i vi(x)(1 − ωi(x)).
    7:     Pˆ(y = j|x) = eˆvj(x)/\sum_{i=0}_N eˆvi(x)
    8: Let y∗ = argmaxj P(y = j|x)
    9: Reject input if y∗ == 0 or P(y = y∗|x) < ǫ
    """
    activations_org, evt_probs_org = activations, evt_probs
    sorted_activations, indices = torch.sort(activations, descending=True, dim=1)
    weights = torch.ones(activations.shape[1])
    weights[:alpha] = torch.arange(1, alpha+1, step=1)
    weights[:alpha] = (alpha-weights[:alpha])/alpha
    weights = 1-(weights[None,:]*evt_probs)
    revisted_activations = sorted_activations * weights
    unknowness_class_prob = torch.sum(sorted_activations * (1-weights), dim=1)
    revisted_activations = torch.gather(revisted_activations, 1, indices)
    probability_vector = torch.cat([unknowness_class_prob[:,None], revisted_activations], dim=1)
    probability_vector = F.softmax(probability_vector, dim=1)

    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    prediction_score[predicted_class == 0] = -1.
    predicted_class = predicted_class-1

    return prediction_score, predicted_class

def magnitude_heuristic(features, gt, score):
    sample_magnitudes = torch.norm(features, p=2, dim=1)
    score, predicted_class = torch.max(score, dim=1)
    return (predicted_class, sample_magnitudes*score)

def proximity_heuristic(features, centers, gt, score):
    distances_to_centers = torch.cdist(features, centers, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
    updated_score = distances_to_centers*score
    score = -1*updated_score
    score, predicted_class = torch.max(score, dim=1)
    return (predicted_class, score)