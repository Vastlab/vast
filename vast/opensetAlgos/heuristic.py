import torch
from torch.nn import functional as F


def openmax_alpha(
    evt_probs, activations, alpha=1, run_paper_version=True, *args, **kwargs
):
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
    6: Define vˆ0(x) = sum_i vi(x)(1 − ωi(x)).
    7:     Pˆ(y = j|x) = eˆvj(x)/sum_{i=0}_N eˆvi(x)
    8: Let y∗ = argmaxj P(y = j|x)
    9: Reject input if y∗ == 0 or P(y = y∗|x) < ǫ
    """
    # convert weibull CDF probabilities from knownness per class to unknownness per class
    per_class_unknownness_prob = 1 - evt_probs

    # Line 1
    sorted_activations, indices = torch.sort(activations, descending=True, dim=1)
    weights = torch.ones(activations.shape[0], activations.shape[1])

    # Line 2-4
    weights[:, :alpha] = torch.arange(1, alpha + 1, step=1)
    if run_paper_version:
        weights[:, :alpha] = (alpha - weights[:, :alpha]) / alpha
    else:
        # The version in the code is slightly different from the algorithm mentioned in the paper
        weights[:, :alpha] = ((alpha + 1) - weights[:, :alpha]) / alpha
    weights[:, :alpha] = 1 - weights[:, :alpha] * torch.gather(
        per_class_unknownness_prob, 1, indices[:, :alpha]
    )

    # Line 5
    revisted_activations = sorted_activations * weights
    # Line 6
    unknowness_class_prob = torch.sum(sorted_activations * (1 - weights), dim=1)
    revisted_activations = torch.scatter(
        torch.ones(revisted_activations.shape), 1, indices, revisted_activations
    )
    probability_vector = torch.cat(
        [unknowness_class_prob[:, None], revisted_activations], dim=1
    )

    # Line 7
    probability_vector = F.softmax(probability_vector, dim=1)
    # Line 8
    prediction_score, predicted_class = torch.max(probability_vector, dim=1)
    # Line 9
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1

    return predicted_class, prediction_score


def magnitude_heuristic(score, features, *args, **kwargs):
    sample_magnitudes = torch.norm(features, p=2, dim=1)
    score, predicted_class = torch.max(score, dim=1)
    return (predicted_class, sample_magnitudes * score)


def proximity_heuristic(score, features, centers, *args, **kwargs):
    distances_to_centers = torch.cdist(
        features, centers, p=2.0, compute_mode="donot_use_mm_for_euclid_dist"
    )
    updated_score = distances_to_centers * score
    score = -1 * updated_score
    score, predicted_class = torch.max(score, dim=1)
    return (predicted_class, score)


def l1_normalization(evt_probs, *args, **kwargs):
    """
    WIP Algo
    """
    any_cls_max_knowness_probability, _ = torch.max(evt_probs, axis=1)
    unknowness_class_prob = 1 - any_cls_max_knowness_probability
    probability_tensor = torch.cat([unknowness_class_prob[:, None], evt_probs], dim=1)
    probability_tensor = (
        probability_tensor / torch.norm(probability_tensor, p=1, dim=1)[:, None]
    )
    prediction_score, predicted_class = torch.max(probability_tensor, dim=1)
    prediction_score[predicted_class == 0] = -1.0
    predicted_class = predicted_class - 1
    return predicted_class, prediction_score
