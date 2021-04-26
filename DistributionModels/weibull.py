import numpy as np
import torch
import pdb

class weibull:
    def __init__(self, saved_model=None,translate_amount_from_zero_for_stability=1):
        self.reversed = False
        self.to_trim = 0
        self.translate_amount_from_zero_for_stability = translate_amount_from_zero_for_stability
        self.smallScoreTensor = torch.tensor(0.0)
        if saved_model:
            self.wbFits = torch.zeros(saved_model['Scale'].shape[0],2)
            self.wbFits[:, 1] = saved_model['Scale']
            self.wbFits[:, 0] = saved_model['Shape']
            self.sign = saved_model['signTensor']
            self._ = saved_model['translate_amount_from_zero_for_stability']
            self.smallScoreTensor = saved_model['smallScoreTensor']
            self.translate_amount_from_zero_for_stability = saved_model['translate_amount_from_zero_for_stability']            
        return

    def tocpu(self):
        self.wbFits = self.wbFits.cpu()
        self.smallScoreTensor = self.smallScoreTensor.cpu()

    def return_all_parameters(self):
        return dict(Scale = self.wbFits[:, 1],
                    Shape = self.wbFits[:, 0],
                    signTensor = self.sign,
                    translate_amount_from_zero_for_stability = self.translate_amount_from_zero_for_stability,
                    smallScoreTensor = self.smallScoreTensor)


    def FitLow(self,data, tailSize, isSorted=False, gpu=0):
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
            self.splits = tailSize//max_tailsize_in_1_chunk + 1
            to_return = self._weibullFilltingInBatches(data, tailSize, isSorted, gpu)
        return to_return


    # fit low reversed  aka fitlow model   does a fit low but instead of having the wscore return the CDF, it returns 1-CDF.  This is not the same as fithigh since it uses the smallest data but then the wscore probability that score is less than the data fit and it is decreasing as it approaches the data..
    
    def FitLowReversed(self,data, tailSize, isSorted=False, gpu=0):
        """
        data --> 5000 weibulls on 0 dim
             --> 10000 distances for each weibull on 1 dim
        """
        to_return = FitLow(self,data, tailSize, isSorted, gpu)
        self.reversed = True        
        return to_return




    # TB's new mode that flips the data around the max and then does a fit low reject so that its effectively modeling values just above the max . 
    def FitHighFlipped(self,data, tailSize, isSorted=False, gpu=0):
        """
        data --> 5000 weibulls on 0 dim
             --> 10000 distances for each weibull on 1 dim
        """
        self.sign = -1
        maxval = data.max()
        #flip the data around the max so the smallest points are just beyond the data but mirrors the distribution
        data = 2* maxval - data
        max_tailsize_in_1_chunk = 100000
        if tailSize <= max_tailsize_in_1_chunk:
            self.splits = 1
            to_return = self._weibullFitting(data, tailSize, isSorted, gpu)
        else:
            self.splits = tailSize//max_tailsize_in_1_chunk + 1
            to_return = self._weibullFilltingInBatches(data, tailSize, isSorted, gpu)
        return to_return

       
    #fit high if trimboth >0, it s the number of items to trim and tailsize is used for for fitting.  For compatabiliwth with first version use in FACTO if  trimboth=0 (or not supplied),, then tailsize is the number items to keep (from smallest)  and trim off all above it.  
    def FitHighTrimmed(self, data, tailsize, isSorted=False, trimboth=0):
        self.sign = 1
        self.splits = 1
#        pdb.set_trace()
        if(trimboth>0):
            self.to_trim = trimboth;
        else: 
            length = list(data.size())[1]
            self.to_trim = length-tailsize;

        return self._weibullFitting(data, tailsize, isSorted)



    def FitHigh(self, data, tailSize, isSorted=False):
        self.sign = 1
        self.splits = 1
        return self._weibullFitting(data, tailSize, isSorted)


    # fit high reversed  aka fithigh model   does a fit high but instead of having the wscore return the CDF, it returns 1-CDF.  This is not the same as fitlow since it uses the largest data but then the wscore probability that score is less than the data fit and it is decreasing as it approaches the data. 

    def FitHighReversed(self, data, tailSize, isSorted=False):
        self.sign = 1
        self.splits = 1
        rval = self._weibullFitting(data, tailSize, isSorted)
        self.reversed = 1
        return rval;



    def wscore(self, distances):
        """
        This function can calculate scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        self.deviceName = distances.device
        scale_tensor = self.wbFits[:,1]
        shape_tensor = self.wbFits[:, 0]
        if self.sign == -1:
            distances = -distances
        if len(distances.shape)==1:
            distances = distances.repeat(shape_tensor.shape[0],1)
        smallScoreTensor=self.smallScoreTensor
        if len(self.smallScoreTensor.shape)==2:
            smallScoreTensor=self.smallScoreTensor[:,0]

        #if translate_amount_from_zero is zero, then we don't use a smallScoreTensor.. (i.e. no shift) and do this as 2 -arm weibull
        if(self.translate_amount_from_zero_for_stability == 0):
            smallScoreTensor=0* smallScoreTensor;
        distances = distances + self.translate_amount_from_zero_for_stability - smallScoreTensor.to(self.deviceName)[None,:]
        weibulls = torch.distributions.weibull.Weibull(scale_tensor.to(self.deviceName),shape_tensor.to(self.deviceName),
                                                       validate_args=False)

        distances = distances.clamp(min=0)
        if(self.reversed):
            return 1-weibulls.cdf(distances)
        else:
            return weibulls.cdf(distances)        


        

    def prob(self, distances):
        """
        This function can calculate raw probability scores from various weibulls for a given set of distances
        :param distances: a 2-D tensor with the number of rows equal to number of samples and number of columns equal to number of weibulls
        Or
        a 1-D tensor with number of elements equal to number of test samples
        :return:
        """
        self.deviceName = distances.device
        scale_tensor = self.wbFits[:,1]
        shape_tensor = self.wbFits[:, 0]
        if self.sign == -1:
            distances = -distances
        if len(distances.shape)==1:
            distances = distances.repeat(shape_tensor.shape[0],1)
        smallScoreTensor=self.smallScoreTensor
        if len(self.smallScoreTensor.shape)==2:
            smallScoreTensor=self.smallScoreTensor[:,0]
        #if translate_amount_from_zero is zero, then we don't use a smallScoreTensor.. (i.e. no shift) and do this as 2 -arm weibull
        if(self.translate_amount_from_zero_for_stability ==0):
            smallScoreTensor=0* smallScoreTensor;
        distances = distances + self.translate_amount_from_zero_for_stability - smallScoreTensor.to(self.deviceName)[None,:]
        weibulls = torch.distributions.weibull.Weibull(scale_tensor.to(self.deviceName),shape_tensor.to(self.deviceName))
        distances = distances.clamp(min=.00000001)  #clamp to avoid blowup and range issues
        return torch.exp(weibulls.log_prob(distances))

    

    def _weibullFitting(self, dataTensor, tailSize, isSorted=False, gpu=0):
        self.deviceName = dataTensor.device
        if self.sign == -1:
            dataTensor = -dataTensor
        
        if self.to_trim >0 :
#            pdb.set_trace()
            len = list(dataTensor.size())[1]
            tsize = min(self.to_trim + tailSize,len);
            #need trimed tail by getting enough data for drop + tailize then  we timmed items by keeping smaller values
            sortedTensor = torch.topk(dataTensor, tsize, dim=1, largest=True, sorted=True).values
            sortedTensor = torch.topk(sortedTensor, tailSize, dim=1, largest=False, sorted=True).values
            smallScoreTensor = sortedTensor[:, 0].unsqueeze(1)            
        else: 
            sortedTensor = torch.topk(dataTensor, tailSize, dim=1, largest=True, sorted=True).values
            smallScoreTensor = sortedTensor[:, tailSize - 1].unsqueeze(1)

        #if translate_amount_from_zero is zero, then we don't use a smallScoreTensor.. (i.e. no shift) and do this as 2 -arm weibull
        if(self.translate_amount_from_zero_for_stability ==0):
            smallScoreTensor=0* smallScoreTensor;            

        processedTensor = sortedTensor + self.translate_amount_from_zero_for_stability - smallScoreTensor
# previously translate_amount_from_zero_for_stability was hard coded at 1. (translateamount from libMR)        
#        processedTensor = sortedTensor + 1 - smallScoreTensor        
        # Returned in the format [Shape,Scale]
        self.smallScoreTensor = smallScoreTensor
        if self.splits == 1:
            self.wbFits = self._fit(processedTensor)
            return self.wbFits,smallScoreTensor
        else:
            return self._fit(processedTensor),smallScoreTensor


    def _weibullFilltingInBatches(self, dataTensor, tailSize, isSorted = False, gpu=0):
        N =  dataTensor.shape[0]
        dtype = dataTensor.dtype
        batchSize = int(np.ceil(N / self.splits))
        resultTensor = torch.zeros(size=(N,2), dtype=dtype)
        reultTensor_smallScoreTensor = torch.zeros(size=(N,1), dtype=dtype)
            
        for batchIter in range(int(self.splits-1)):
          startIndex = batchIter*batchSize
          endIndex = startIndex + batchSize
          data_batch = dataTensor[startIndex:endIndex,:].cuda(gpu)
          result_batch, result_batch_smallScoreTensor = self._weibullFitting(data_batch, tailSize, isSorted)
          resultTensor[startIndex:endIndex,:] = result_batch.cpu()
          reultTensor_smallScoreTensor[startIndex:endIndex,:] = result_batch_smallScoreTensor.cpu()
          
        # process the left-over
        startIndex = (self.splits-1)*batchSize
        endIndex = N
          
        data_batch = dataTensor[startIndex:endIndex,:].cuda(gpu)
        result_batch, result_batch_smallScoreTensor = self._weibullFitting(data_batch, tailSize, isSorted)
        resultTensor[startIndex:endIndex,:] = result_batch.cpu()
        reultTensor_smallScoreTensor[startIndex:endIndex,:] = result_batch_smallScoreTensor.cpu()
    
        self.wbFits = resultTensor
        self.smallScoreTensor = reultTensor_smallScoreTensor


    def _fit(self, data, iters=100, eps=1e-6):
        """
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
        computed_params = torch.zeros(data.shape[0],2).double().to(self.deviceName)
        not_completed = torch.ones(data.shape[0], dtype=torch.bool).to(self.deviceName)
        for t in range(iters):
            if torch.all(torch.logical_not(not_completed)):
                break
            # Partial derivative df/dk
            x_k = data ** torch.transpose(k.repeat(data.shape[1],1),0,1)
            x_k_ln_x = x_k * ln_x
            fg = torch.sum(x_k,dim=1)
            del x_k
            ff = torch.sum(x_k_ln_x,dim=1)
            ff_prime = torch.sum(x_k_ln_x * ln_x,dim=1)
            del x_k_ln_x
            ff_by_fg = ff/fg
            del ff
            f = ff_by_fg - torch.mean(ln_x,dim=1) - (1.0 / k)
            f_prime = (ff_prime / fg - (ff_by_fg**2)) + (1. / (k * k))
            del ff_prime, fg
            # Newton-Raphson method k = k - f(k;x)/f'(k;x)
            lastk = k
            k -= f / f_prime
            if(torch.isnan(k)):
                pdb.set_trace()
            computed_params[not_completed*torch.isnan(f),:] = torch.tensor([float('nan'),float('nan')]).double().to(self.deviceName)
            not_completed[abs(k - k_t_1) < eps] = False
            computed_params[torch.logical_not(not_completed),0] = k[torch.logical_not(not_completed)]
            lam = torch.mean(data ** torch.transpose(k.repeat(data.shape[1],1),0,1),dim=1) ** (1.0 / k)
            # Lambda (scale) can be calculated directly
            computed_params[torch.logical_not(not_completed), 1] = lam[torch.logical_not(not_completed)]
            k_t_1 = k.clone()
        return computed_params  # Shape (SC), Scale (FE)
