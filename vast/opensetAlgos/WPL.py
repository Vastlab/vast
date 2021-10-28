"""
Author: Akshay Raj Dhamija

Weibull Prototype Learning (WPL)

There is no paper
"""
import torch
import itertools
from collections import OrderedDict
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from typing import Iterator, Tuple, List, Dict
from typing import OrderedDict as OrderedDict_typing


def WPL_Params(parser):
  raise NotImplementedError("Akshay Raj Dhamija will complete this function later")


def fit_high(distances, distance_multiplier, tailsize):
    if tailsize <= 1:
        tailsize = min(tailsize * distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize, distances.shape[1]))
    mr = weibull.weibull()
    mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr

  
def WPL_Training(
    pos_classes_to_process: List[str],
    features_all_classes: OrderedDict_typing[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process class.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this ordered dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the WPL_Params function above.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    """
    with torch.no_grad():
        for pos_cls_name in pos_classes_to_process:
            features = features_all_classes[pos_cls_name].clone().to(f"cuda:{gpu}")
            assert args.dimension == features.shape[1]

            center = torch.mean(features, dim=0).to(f"cuda:{gpu}")
            distances = 
            for tailsize, distance_multiplier in itertools.product(
                args.tailsize, args.distance_multiplier
            ):
                weibull_list = list()
                for _ in args.dimension:
                  weibull_model = fit_high(distances.T, distance_multiplier, tailsize)
                  weibull_list.append(weibull_model)

                  yield (
                    f"TS_{org_tailsize}_DM_{distance_multiplier:.2f}_CT_{cover_threshold:.2f}",
                    (pos_cls_name, 
                     OrderedDict([('center', center), ('weibull_list', weibull_list))]),
                  )
                  
              
              
              
              
              
              
              
            

