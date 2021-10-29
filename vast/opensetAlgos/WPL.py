
"""
Motivated from: https://github.com/Vastlab/vast/blob/main/vast/opensetAlgos/openmax.py

Weibull Prototype Learning (WPL)


WPL fits a Weibull distribution for each dimension of the feature of each class. 
If we have 'n' classes and the dimension of the feature is 'd', then it fit 'n x d' Weibull. 
It computes and saves the center (mean) of each class, which is equivalent to Prototype in the literature. 
So, each Prototype has 'd' dimensions. Therefore, we fit 'd' Weibulls for each center (Prototype).
Distance of a point (an input) is defined as the absolute value of the difference between 
the point (input) and the center (Prototype). So, distance is a vector with 'd' dimension. 
Distance is not a scalar. The probability of each class is the minimum Weibull probability of all dimensions.

In the training time, if a Weibll (class, dimension) has 5 or more unique values, maximum likelihood 
estimator (MLE)  or maximum a posterior (MAP) can be used to estimate the Weibull parameter. 
If a Weibll (class, dimension) has less than 5 unique values, instead of estimating, it uses 
the default parameter for creating Weibull. The default parameters are the default scale and default shape.


WPL has 5 arguments: dimension, tail size, distance multiplier, default_scale, default shape

The main difference between WPL and OpenMax is how they compute the distance. 
Therefore, the number of Weibull is different between WPL and OpenMax. Another difference 
is how they are training Weibull when the number of unique values is less than 5.


"""

import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from typing import Iterator, Tuple, List, Dict, OrderedDict


def WPL_Params(parser):
    WPL_params = parser.add_argument_group("WPL params")
    WPL_params.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use default: %(default)s",
    )
    WPL_params.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[1.0],
        help="distance multiplier to use default: %(default)s",
    )
    WPL_params.add_argument(
        "--default_scale",
        nargs="+",
        type=float,
        default=[1.0],
        help="Weibull scale to use when the number of uniqe element is less than 5 default: %(default)s",
    )
    WPL_params.add_argument(
        "--default_shape",
        nargs="+",
        type=float,
        default=[0.1],
        help="Weibull shape to use when the number of uniqe element is less than 5 default:  %(default)s",
    )
    return parser, dict(
        group_parser=WPL_params,
        param_names=("tailsize", "distance_multiplier", "default_scale", "default_shape"),
        param_id_string="TS_{}_DM_{:.4f}_SC_{:.4f}_SH_{:.4f}",
    )


def fit_high(distances, distance_multiplier, tailsize, default_shape, default_scale):
    distances = torch.unique(distances)
    if tailsize <= 1:
        tailsize = min(tailsize * distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize, distances.shape[1]))
    mr = weibull.weibull()
    if distances.shape[1] < 5:
        pass
    else:
        mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr

  
def WPL_Training(
    pos_classes_to_process: List[str],
    features_all_classes: OrderedDict[str, torch.Tensor],
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
            distances = torch.abs(features - center.view(1,args.dimension).repeat(features.shape[0], 1))
            for tailsize, distance_multiplier, default_shape, default_scale  in itertools.product(
                args.tailsize, args.distance_multiplier, args.default_shape, args.default_scale
            ):
                  weibull_list = list()
                  for k in args.dimension:
                      weibull_model = fit_high(distances[:,k].T, distance_multiplier, tailsize, default_shape, default_scale)
                      weibull_list.append(weibull_model)
    
                      yield (
                        f"TS_{tailsize}_DM_{distance_multiplier:.4f}_SC{default_scale:.4f}_SH_{default_shape:.4f}",
                        (pos_cls_name,  {'center':center, 'weibull_list': weibull_list})
                      )
                      
              
              
              
def WPL_Inference(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models: Dict = None,
) -> Iterator[Tuple[str, Tuple[str, torch.Tensor]]]:
    """
    :param pos_classes_to_process: List of batches to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of
                                the keys for this dictionary
    :param args: Can be a named tuple or an argument parser object containing the arguments mentioned in the WPL_Params
                function above. Only the distance_metric argument is actually used during inferencing.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: The collated model created for a single hyper parameter combination.
    :return: Iterator(Tuple(str, Tuple(batch_identifier, torch.Tensor)))
    """         
    for batch_to_process in pos_classes_to_process:
        test_cls_feature = features_all_classes[batch_to_process].to(f"cuda:{gpu}")
        assert test_cls_feature.shape[0] != 0
        probs = []
        for cls_no, cls_name in enumerate(models.keys())     
            center = model[cls_name]["center"].to(f"cuda:{gpu}")
            distances = torch.abs(test_cls_feature - center.view(1,args.dimension).repeat(test_cls_feature.shape[0], 1))
            weibull_list = models[class_name]["weibull_list"]
            p = torch.empty(args.dimension)
            for k in args.dimension:
                weibull = weibull_list[k]
                p[k] =  1 - weibull.wscore(distances[:,k].cpu()) )
            probs.append(  torch.min(p)  )
              
        probs = torch.cat(probs, dim=1)
        yield ("probs", (batch_to_process, probs))
              
            

