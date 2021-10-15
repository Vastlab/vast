"""This is a simple automated test of the extreme value machine wrapper to
ensure it functions as desired. The toy task is a simple simulation of four 2D
gaussians whose locations are centered on the edge of the unit circle of this
space, along the axes of this space, that is centered at coordinates [0,1],
[1,0], [0,-1], and [-1,0]. These gaussians form the distributions of samples
from four different classes to be classified. The classification task is given
the 2D coordinates, classify that point with a probability vector to specify
which class it most likely belongs. Naturally, being evenly spaced from one
another on the unit circle in this 2D space, the coordinates near [0,0] are the
most uncertain and are expected to produce the closed classification
probability vector of [0.25, 0.25, 0.25, 0.25] at coordinates [0,0]. This toy
problem allows for basic assessment of the ExtremeValueMachine class to ensure
it can at least fit, predict, save, and load etc., and possibly be used for
diagnosis of performance of the EVM.
"""
import torch


class ToyClassify2D4Gaussians(object):
    """Create the generative sampling procedure for obtaining coordinates of
    points as feature data along with the label of which Gaussian distribution
    they belong to.
    """
    def __init__(self, locs=None, scales=0.5, labels=None):
        # TODO create PyTorch Gaussian Distributions at locs and scales
        if locs is None:
            locs =  [[1, 0], [0, 1], [-1, 0], [0, -1]]

        if not isinstance(scales, list):
            scales = [scales] * len(locs)

        self.gaussians = [
            torch.distributions.multivariate_normal.MultivariateNormal(
                torch.Tensor(loc),
                torch.eye(2) * scales[i],
            )
            for i, loc in enumerate(locs)
        ]

    def sample(num):
        """Uniformly samples from all Gaussian Distributions returning the
        coordinates and label of their source distribution.
        """
        return coords, label

    def sample_n(num):
        """Uniformly samples from all Gaussian Distributions returning the
        coordinates and label of their source distribution.
        """

    def equal_sample_n(num):
        """Equally samples from all Gaussian Distributions `num` times each
        returning the coordinates and label of their source distribution.
        """

class TestEVMToyClassify2D4Gaussians(object):
    """PyTest unit tests using the toy classification simulation."""
    def __init__(self):
        raise NotImplementedError()
        # TODO create toy simulation

        # TODO create ExtremeValueMachine

    def init_fit_extreme_value_machine(self):
        raise NotImplementedError()

    def predict_extreme_value_machine(self):
        raise NotImplementedError()

    def increment_fit_extreme_value_machine(self):
        raise NotImplementedError()

    def predict_increment_extreme_value_machine(self):
        raise NotImplementedError()
