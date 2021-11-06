"""
Per Dimension Weibull
Warning: While this wrapper has been written for ease of use, for the purpose of multiprocessing it may add unnecessary
overhead both in terms of memory usage and compute.
While the multiprocessing for training runs multiple dimensions parallely on different GPUs,
during inferencing it runs multiple classes on different GPUs
"""

import argparse
from typing import Iterator, Tuple, List, Dict
import torch
from vast import opensetAlgos
from vast.tools import pairwisedistances
from vast import tools

def PDW_Params(parser):
    PDW_params = parser.add_argument_group("PDW params")
    PDW_params.add_argument(
        "--set_shape_to",
        type=float,
        default=1.0,
        help="Set shape to this value if none could be computed : %(default)s",
    )
    PDW_params.add_argument(
        "--set_scale_to",
        type=float,
        default=1.0,
        help="Set scale to this value if none could be computed : %(default)s",
    )
    PDW_params.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use default: %(default)s",
    )
    PDW_params.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[1.0],
        help="distance multiplier to use default: %(default)s",
    )
    PDW_params.add_argument(
        "--distance_metric",
        default="cosine",
        type=str,
        choices=list(pairwisedistances.__dict__.keys()),
        help="distance metric to use default: %(default)s",
    )
    PDW_params.add_argument(
        "--distances_unique",
        action="store_true",
        default=False,
        help="Use unique distances during fitting",
    )
    PDW_params.add_argument(
        '--Algo_choice',
        default='OpenMax',
        type=str,
        choices=['OpenMax','EVM','Turbo_EVM','MultiModalOpenMax'],
        help='Name of the openset detection algorithm'
    )
    return parser, dict(
        group_parser=PDW_Params,
        param_names=("tailsize", "distance_multiplier"),
        param_id_string="TS_{}_DM_{:.2f}",
    )

    """
    if "OOD_Algo" not in known_args.__dict__ and "OOD_Algo" not in unknown_args.__dict__:
        known_args.OOD_Algo="OpenMax"
    else:
        if "OOD_Algo" in known_args.__dict__:
            known_args.OOD_Algo = known_args.OOD_Algo
        else:
            known_args.OOD_Algo = unknown_args.OOD_Algo

    # Adding Algorithm Params
    params_parser = argparse.ArgumentParser(parents = [PDW_params], formatter_class = argparse.RawTextHelpFormatter)
    parser_to_return, algo_params = getattr(opensetAlgos, known_args.OOD_Algo + '_Params')(params_parser)

    return parser_to_return, algo_params
    """

def PDW_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None
):
    if "Algo_choice" not in args.__dict__:
        args.Algo_choice="Algo_choice"
    if "world_size" not in args.__dict__:
        args.world_size=1
    args.distances_unique = True

    # Convert dict from based on classes to based on dimension, making each dimension a class
    features_all_classes, class_identifiers = tools.features_to_dim(features_all_classes, classes_to_process=None)

    OOD_Method = getattr(opensetAlgos, f'{args.Algo_choice}_Training')
    pos_classes_to_process = list(features_all_classes.keys())
    div, mod = divmod(len(pos_classes_to_process), args.world_size)
    pos_classes_to_process = pos_classes_to_process[gpu * div + min(gpu, mod):(gpu + 1) * div + min(gpu + 1, mod)]
    print(f"Processing classes {pos_classes_to_process}")

    algo_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
    for output in algo_iterator:
        yield output

def PDW_Inference(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None
):
    if "Algo_choice" not in args.__dict__:
        args.Algo_choice="OpenMax"
    args.distances_unique = True

    # Convert dict from based on classes to based on dimension, making each dimension a class
    features_all_classes, class_identifiers = tools.features_to_dim(features_all_classes,
                                                                    classes_to_process=pos_classes_to_process)

    pos_classes_to_process = list(features_all_classes.keys())
    print(f"Processing classes {pos_classes_to_process}")

    # Set default shape scale of weibull model if none could be computed
    models = opensetAlgos.heuristic.set_shape_scale_defaults(models,
                                                             set_shape_to=args.set_shape_to,
                                                             set_scale_to=args.set_scale_to)

    OOD_Method = getattr(opensetAlgos, f'{args.Algo_choice}_Inference')
    algo_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
    all_class_results = []
    for output in algo_iterator:
        string, (batch_to_process, probs) = output
        # The rest of index computations were redundant we only need the class we are currently processing
        probs = probs[:, int(batch_to_process)]
        all_class_results.append(probs)
    all_class_results = torch.stack(all_class_results, dim=1)
    for cls in set(class_identifiers.tolist()):
        probs = all_class_results[class_identifiers==cls,:]
        yield ("probs", (cls, probs))