"""
Author: Terrance E. Boult.
This class extends the default libmr and aims to provide more useful curve fitting options.
"""
import torch
from .libmr import libmr


class weibull(libmr):
    # TB's new mode that flips the data around the max and then does a fit low reject so that its effectively modeling just above the max .
    def FitHighFlipped(self, data, tailSize, isSorted=False, gpu=0):
        maxval = data.max(dim=1).values
        # flip the data around the max so the smallest points are just beyond the data but mirrors the distribution
        data = 2 * maxval - data
        return self.FitLow(self, data, tailSize, isSorted=isSorted, gpu=gpu)

    def FitLowReversed(self, data, tailSize, isSorted=False, gpu=0):
        self.reversed = True
        return self.FitLow(self, data, tailSize, isSorted=isSorted, gpu=gpu)

    def prob(self, distances):
        """
        This function can calculate raw probability scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        weibulls, distances = self.compute_weibull_object(distances)
        return torch.exp(weibulls.log_prob(distances))
