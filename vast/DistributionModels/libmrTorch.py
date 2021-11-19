"""
Authors: Akshay Raj Dhamija and Touqeer Ahmad.
This is gpu based reimplementation of libmr.
"""
import numpy as np
import torch


class libmr:
    def __init__(self, saved_model=None, translateAmount=1):
        self.translateAmount = translateAmount
        if saved_model:
            self.wbFits = torch.zeros(saved_model["Scale"].shape[0], 2)
            self.wbFits[:, 1] = saved_model["Scale"]
            self.wbFits[:, 0] = saved_model["Shape"]
            self.sign = saved_model["signTensor"]
            self.translateAmount = saved_model["translateAmountTensor"]
            self.smallScoreTensor = saved_model["smallScoreTensor"]
        return

    def tocpu(self):
        self.wbFits = self.wbFits.cpu()
        self.smallScoreTensor = self.smallScoreTensor.cpu()

    def return_all_parameters(self):
        return dict(
            Scale=self.wbFits[:, 1],
            Shape=self.wbFits[:, 0],
            signTensor=self.sign,
            translateAmountTensor=self.translateAmount,
            smallScoreTensor=self.smallScoreTensor,
        )

    def FitLow(self, data, tailSize, isSorted=False, gpu=0):
        """
        data --> 5000 weibulls on 0 dim
             --> 10000 distances for each weibull on 1 dim
        """
        self.sign = -1
        max_tailsize_in_1_chunk = 100000
        if tailSize <= max_tailsize_in_1_chunk:
            self.splits = 1
            to_return = self._weibullFitting(data, tailSize, isSorted, gpu)
        else:
            self.splits = tailSize // max_tailsize_in_1_chunk + 1
            to_return = self._weibullFilltingInBatches(data, tailSize, isSorted, gpu)
        return to_return

    def FitHigh(self, data, tailSize, isSorted=False, gpu=0):
        self.sign = 1
        self.splits = 1
        return self._weibullFitting(data, tailSize, isSorted, gpu)

    def compute_weibull_object(self, distances):
        self.deviceName = distances.device
        scale_tensor = self.wbFits[:, 1]
        shape_tensor = self.wbFits[:, 0]
        if self.sign == -1:
            distances = -distances
        if len(distances.shape) == 1:
            distances = distances.repeat(shape_tensor.shape[0], 1)
        smallScoreTensor = self.smallScoreTensor
        if len(self.smallScoreTensor.shape) == 2:
            smallScoreTensor = self.smallScoreTensor[:, 0]
        distances = (
            distances
            + self.translateAmount
            - smallScoreTensor.to(self.deviceName)[None, :]
        )
        weibulls = torch.distributions.weibull.Weibull(
            scale_tensor.to(self.deviceName),
            shape_tensor.to(self.deviceName),
            validate_args=False,
        )
        distances = distances.clamp(min=0)
        return weibulls, distances

    def wscore(self, distances, isReversed=False):
        """
        This function can calculate scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        weibulls, distances = self.compute_weibull_object(distances)
        if isReversed:
            return 1 - weibulls.cdf(distances)
        else:
            return weibulls.cdf(distances)

    def _weibullFitting(self, dataTensor, tailSize, isSorted=False, gpu=0):
        self.deviceName = dataTensor.device
        if isSorted:
            sortedTensor = dataTensor
        else:
            if self.sign == -1:
                dataTensor = -dataTensor
            sortedTensor = torch.topk(
                dataTensor, tailSize, dim=1, largest=True, sorted=True
            ).values

        smallScoreTensor = sortedTensor[:, tailSize - 1].unsqueeze(1)
        processedTensor = sortedTensor + self.translateAmount - smallScoreTensor
        # Returned in the format [Shape,Scale]
        wbFits = self._fit(processedTensor)
        if self.splits == 1:
            self.wbFits = wbFits
            self.smallScoreTensor = smallScoreTensor
        return wbFits, smallScoreTensor

    def _weibullFilltingInBatches(self, dataTensor, tailSize, isSorted=False, gpu=0):
        N = dataTensor.shape[0]
        dtype = dataTensor.dtype
        batchSize = int(np.ceil(N / self.splits))
        resultTensor = torch.zeros(size=(N, 2), dtype=dtype)
        reultTensor_smallScoreTensor = torch.zeros(size=(N, 1), dtype=dtype)
        for batchIter in range(int(self.splits - 1)):
            startIndex = batchIter * batchSize
            endIndex = startIndex + batchSize
            data_batch = dataTensor[startIndex:endIndex, :].cuda(gpu)
            result_batch, result_batch_smallScoreTensor = self._weibullFitting(
                data_batch, tailSize, isSorted
            )
            resultTensor[startIndex:endIndex, :] = result_batch.cpu()
            reultTensor_smallScoreTensor[
                startIndex:endIndex, :
            ] = result_batch_smallScoreTensor.cpu()

        # process the left-over
        startIndex = (self.splits - 1) * batchSize
        endIndex = N

        data_batch = dataTensor[startIndex:endIndex, :].cuda(gpu)
        result_batch, result_batch_smallScoreTensor = self._weibullFitting(
            data_batch, tailSize, isSorted
        )
        resultTensor[startIndex:endIndex, :] = result_batch.cpu()
        reultTensor_smallScoreTensor[
            startIndex:endIndex, :
        ] = result_batch_smallScoreTensor.cpu()

        self.wbFits = resultTensor
        self.smallScoreTensor = reultTensor_smallScoreTensor

    def _fit(self, data, iters=100, eps=1e-6):
        """
        Adapted from: https://github.com/mlosch/python-weibullfit/blob/0fc6fbe5103c5a2e3ac3374433978f0b816b70be/weibull/backend_pytorch.py#L5
        Adds functionality to fit multiple weibull models in a single tensor using 2D torch tensors.
        Fits multiple 2-parameter Weibull distributions to the given data using maximum-likelihood estimation.
        :param data: 2d-tensor of samples. Each value must satisfy x > 0.
        :param iters: Maximum number of iterations
        :param eps: Stopping criterion. Fit is stopped ff the change within two iterations is smaller than eps.
        :return: tensor with first column Shape, and second Scale these can be (NaN, NaN) if a fit is impossible.
            Impossible fits may be due to 0-values in data.
        """
        k = torch.ones(data.shape[0]).double().to(self.deviceName)
        k_t_1 = k.clone()
        ln_x = torch.log(data)
        computed_params = torch.zeros(data.shape[0], 2).double().to(self.deviceName)
        not_completed = torch.ones(data.shape[0], dtype=torch.bool).to(self.deviceName)
        for t in range(iters):
            if torch.all(torch.logical_not(not_completed)):
                break
            # Partial derivative df/dk
            x_k = data ** torch.transpose(k.repeat(data.shape[1], 1), 0, 1)
            x_k_ln_x = x_k * ln_x
            fg = torch.sum(x_k, dim=1)
            del x_k
            ff = torch.sum(x_k_ln_x, dim=1)
            ff_prime = torch.sum(x_k_ln_x * ln_x, dim=1)
            del x_k_ln_x
            ff_by_fg = ff / fg
            del ff
            f = ff_by_fg - torch.mean(ln_x, dim=1) - (1.0 / k)
            f_prime = (ff_prime / fg - (ff_by_fg ** 2)) + (1.0 / (k * k))
            del ff_prime, fg
            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            k -= f / f_prime
            computed_params[not_completed * torch.isnan(f), :] = (
                torch.tensor([float("nan"), float("nan")]).double().to(self.deviceName)
            )
            not_completed[abs(k - k_t_1) < eps] = False
            computed_params[torch.logical_not(not_completed), 0] = k[
                torch.logical_not(not_completed)
            ]
            lam = torch.mean(
                data ** torch.transpose(k.repeat(data.shape[1], 1), 0, 1), dim=1
            ) ** (1.0 / k)
            # Lambda (scale) can be calculated directly
            computed_params[torch.logical_not(not_completed), 1] = lam[
                torch.logical_not(not_completed)
            ]
            k_t_1 = k.clone()
        return computed_params  # Shape (SC), Scale (FE)
