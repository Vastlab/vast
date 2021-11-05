"""
Per Dimension Weibull
Warning: While this wrapper has been written for ease of use, for the purpose of multiprocessing it may add unnecessary
overhead both in terms of memory usage and compute.
"""

from vast import opensetAlgos
from vast import tools

def PDW_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None
):
    if "OOD_Algo" not in args.__dict__:
        args.OOD_Algo="OpenMax"
    args.distances_unique = True
    OOD_Method = getattr(vast.opensetAlgos, f'{args.OOD_Algo}_Training')

    # Convert dict from based on classes to based on dimension, making each dimension a class
    features_all_classes = tools.features_to_dim(features_all_classes)

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
    if "OOD_Algo" not in args.__dict__:
        args.OOD_Algo="OpenMax"
    args.distances_unique = True
    OOD_Method = getattr(vast.opensetAlgos, f'{args.OOD_Algo}_Inference')

    # Convert dict from based on classes to based on dimension, making each dimension a class
    features_all_classes = tools.features_to_dim(features_all_classes)

    # Set default shape scale of weibull model if none could be computed
    models = heuristic.set_shape_scale_defaults(models,
                                                set_shape_to=args.set_shape_to,
                                                set_scale_to=args.set_scale_to)

    algo_iterator = OOD_Method(pos_classes_to_process, features_all_classes, args, gpu, models)
    for output in algo_iterator:
        string, (batch_to_process, probs) = output
        # The rest of index computations were redundant we only need the class we are currently processing
        probs = probs[:, int(batch_to_process)][:, None]
        output = (string, (batch_to_process, probs))
        yield output