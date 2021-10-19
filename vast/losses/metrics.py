import torch

"""This file contains different metrics that can be applied to evaluate the training"""


def accuracy(prediction, target):
    """Computes the classification accuracy of the classifier based on known samples only.
    Any target that does not belong to a certain class (target is -1) is disregarded.

    Parameters:

      prediction: the output of the network, can be logits or softmax scores

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      correct: The number of correctly classified samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        total = torch.sum(known, dtype=int)
        if total:
            correct = torch.sum(
                torch.max(prediction[known], axis=1).indices == target[known], dtype=int
            )
        else:
            correct = 0

    return torch.tensor((correct, total))


def sphere(representation, target, sphere_radius=None):
    """Computes the radius of unknown samples.
    For known samples, the radius is computed and added only when sphere_radius is not None.

    Parameters:

      representation: the feature vector of the samples

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      length: The sum of the length of the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        magnitude = torch.norm(representation, p=2, dim=1)

        sum = torch.sum(magnitude[~known])
        total = torch.sum(~known)

        if sphere_radius is not None:
            sum += torch.sum(torch.clamp(sphere_radius - magnitude, min=0.0))
            total += torch.sum(known)

    return torch.tensor((sum, total))


def confidence(logits, target, negative_offset=0.1):
    """Measures the softmax confidence of the correct class for known samples,
    and 1 + negative_offset - max(confidence) for unknown samples.

    Parameters:

      logits: the output of the network, must be logits

      target: the vector of true classes; can be -1 for unknown samples

    Returns a tensor with two entries:

      confidence: the sum of the confidence values for the samples

      total: The total number of considered samples
    """

    with torch.no_grad():
        known = target >= 0

        pred = torch.nn.functional.softmax(logits, dim=1)
        #    import ipdb; ipdb.set_trace()

        confidence = 0.0
        if torch.sum(known):
            confidence += torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            confidence += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            )

    return torch.tensor((confidence, len(logits)))
