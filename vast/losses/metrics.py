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

        confidence = 0.0
        if torch.sum(known):
            confidence += torch.sum(pred[known, target[known]])
        if torch.sum(~known):
            confidence += torch.sum(
                1.0 + negative_offset - torch.max(pred[~known], dim=1)[0]
            )

    return torch.tensor((confidence, len(logits)))


def split_confidence(logits, target, negative_offset = 0., unknown_class=-1):
    """Measures the softmax confidence of the correct class for known samples and for unknown samples:

    * with unknown_class = -1: 1 + negative_offset - max(confidence)
    * with unknown_class =  C: 1 - max(confidence[:C]) for unknown samples

    Parameters:

        logits: the output of the network, must be logits

        target: the vector of true classes; can be -1 for unknown samples

        negative_offset: the value to be added to the unknown confidence to turn the maximum to one, usually 1/C with C being the number of classes

        unknown_class: The class index that should be considered the unknown class; can be -1 or C

    Returns a tuple with four entries:

        known_confidence: the sum of the confidence values for the known samples

        unknown_confidence: the sum of the confidence values for the unknown samples

        known_samples: The total number of considered known samples in this batch

        unknown_samples: The total number of considered unknown samples in this batch
    """

    with torch.no_grad():
        known = target != unknown_class

        pred = torch.nn.functional.softmax(logits, dim=1)

        known_confidence = 0.
        unknown_confidence = 0.
        if torch.sum(known):
            known_confidence = torch.sum(pred[known,target[known]])
        if torch.sum(~known):
            if unknown_class == -1:
                unknown_confidence = torch.sum(1. + negative_offset - torch.max(pred[~known], dim=1)[0])
            else:
                unknown_confidence = torch.sum(1. - torch.max(pred[~known, :unknown_class], dim=1)[0])

    return known_confidence, unknown_confidence, torch.sum(known), torch.sum(~known)



def auc_split(logits, target, unknown_class=-1):
    """Computes the scores such that they later can be used to compute an ROC curve.
    After collecting data from all samples, you can call ``sklearn.metrics.roc_auc_score(scores, labels)`` to determine the quality of your current network (larger values are better).

    The scores for the known class will be softmax output for the corresponding class.
    The scores for the unknown class will be the maximum softmax output over all known classes.

    Parameters:

        logits: the output of the network, must be logits

        target: the vector of true classes; can be -1 for unknown samples

        unknown_class: The class index that should be considered the unknown class; can be -1 or C

    Returns:

        scores: A list of scores for the samples.

        labels: A list of binary labels (known/unknown) for the samples.
    """

    with torch.no_grad():
        known = target == unknown_class
        unknown = ~known
        last = unknown_class if unknown_class >= 0 else None

        pred = torch.nn.functional.softmax(logits, dim=1)

        scores = torch.empty(len(target))
        labels = torch.empty(len(target))

        if torch.sum(known):
            scores[known] = pred[known, target[known]]
            labels[known] = 1

        if torch.sum(unknown):
            scores[unknown] = torch.max(pred[unknown, :last], dim=1)
            labels[unknown] = -1

    return scores.tolist(), labels.tolist()
