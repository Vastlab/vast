import torch

"""This file contains different metrics that can be applied to evaluate the training"""

def accuracy(prediction, target):
  """Computes the classification accuracy of the classifier based on known samples only.
  Any target that does not belong to a certain class (where its maximum value is not 1) is disregarded.

  Parameters:

    prediction: the output of the network, can be logits or softmax scores

    target: the vector of true classes; can be -1 for unknown samples

  Returns a tensor with two entries:

    correct: The number of correctly classified samples

    total: The total number of considered samples
  """

  with torch.no_grad():
    known = target >=0

    total = torch.sum(known, dtype=int)
    if total:
      correct = torch.sum(torch.max(prediction[known], axis=1).indices == target[known], dtype=int)
    else:
      correct = 0

  return torch.tensor((correct, total))


def sphere(representation, target, sphere_radius = None):
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
    known = target >=0

    magnitude = torch.norm(representation, p=2, dim=1)

    sum = torch.sum(magnitude[~known])
    total = torch.sum(~known)

    if sphere_radius is not None:
      sum += torch.sum(torch.clamp(sphere_radius - magnitude, min=0.))
      total += torch.sum(knwon)


  return torch.tensor((sum, total))
