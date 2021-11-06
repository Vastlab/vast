"""
Author: Akshay Raj Dhamija

@inproceedings{bendale2016towards,
  title={Towards open set deep networks},
  author={Bendale, Abhijit and Boult, Terrance E},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1563--1572},
  year={2016}
}

While the reimplementation in this file only performs the EVT-recognition part of the original paper, for the actual
openmax algorithm use the functionality from this file in conjunction with openmax_alpha function in heuristic.py
"""
import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from typing import Iterator, Tuple, List, Dict


def OpenMax_Params(parser):
    OpenMax_params = parser.add_argument_group("OpenMax params")
    OpenMax_params.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[1.0],
        help="distance multiplier to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--distance_metric",
        default="cosine",
        type=str,
        choices=list(pairwisedistances.implemented_distances),
        help="distance metric to use default: %(default)s",
    )
    OpenMax_params.add_argument(
        "--distances_unique",
        action="store_true",
        default=False,
        help="Use unique distances during fitting",
    )
    return parser, dict(
        group_parser=OpenMax_params,
        param_names=("tailsize", "distance_multiplier"),
        param_id_string="TS_{}_DM_{:.2f}",
    )


def fit_high(distances, distance_multiplier, tailsize):
    if tailsize <= 1:
        tailsize = min(tailsize * distances.shape[1], distances.shape[1])
    tailsize = int(min(tailsize, distances.shape[1]))
    if distances.shape[1] < 5:
        mr = weibull.weibull(
                dict(
                    Scale=torch.Tensor([-1]),
                    Shape=torch.Tensor([-1]),
                    translateAmountTensor=torch.Tensor([0.0]),               
                    signTensor= 1,
                    smallScoreTensor=torch.Tensor([0.0]),
                )
            )
    else:
        mr = weibull.weibull()
        mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr


def OpenMax_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the EVM_Params function.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    """
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name].clone().to(f"cuda:{gpu}")
        MAV = torch.mean(features, dim=0).to(f"cuda:{gpu}")
        distances = pairwisedistances.__dict__[args.distance_metric](
            features, MAV[None, :]
        )
        for tailsize, distance_multiplier in itertools.product(
            args.tailsize, args.distance_multiplier
        ):
            # check if unique distances are desired
            if args.distances_unique:
                distances = torch.unique(distances)[:, None]

            weibull_model = fit_high(distances.T, distance_multiplier, tailsize)
            yield (
                f"TS_{tailsize}_DM_{distance_multiplier:.2f}",
                (pos_cls_name, dict(MAV=MAV.cpu()[None, :], weibulls=weibull_model)),
            )


def OpenMax_Inference(
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
    :param args: Can be a named tuple or an argument parser object containing the arguments mentioned in the EVM_Params
                function above. Only the distance_metric argument is actually used during inferencing.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: The collated model created for a single hyper parameter combination.
    :return: Iterator(Tuple(str, Tuple(batch_identifier, torch.Tensor)))
    """
    for batch_to_process in pos_classes_to_process:
        features = features_all_classes[batch_to_process].to(f"cuda:{gpu}")
        probs = []
        for class_name in sorted(models.keys()):
            MAV = models[class_name]["MAV"].to(f"cuda:{gpu}")
            distances = pairwisedistances.__dict__[args.distance_metric](features, MAV)
            probs.append(1 - models[class_name]["weibulls"].wscore(distances.cpu()))
        probs = torch.cat(probs, dim=1)
        yield ("probs", (batch_to_process, probs))
