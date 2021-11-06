"""
Author: Akshay Raj Dhamija.
@MSthesis{dhamija2018openset,
  title={An Openset Approach To Object Detection},
  author={Dhamija, Akshay Raj},
  year={2018},
  school={University of Colorado Colorado Springs}
}

This implementation supports various clustering algorithms like KMeans, DBScan and FINCH.
Compute time for both KMeans and DBScan has been reduced by using GPU based implementations provided in
vast/clusteringAlgos/clustering.py
"""
import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from .openmax import fit_high
from typing import Iterator, Tuple, List, Dict


def MultiModalOpenMax_Params(parser):
    MultiModalOpenMax_params = parser.add_argument_group("MultiModalOpenMax params")
    MultiModalOpenMax_params.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use default: %(default)s",
    )
    MultiModalOpenMax_params.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[1.0],
        help="distance multiplier to use default: %(default)s",
    )
    MultiModalOpenMax_params.add_argument(
        "--translateAmount",
        nargs="+",
        type=float,
        default=1.0,
        help="translateAmount to use default: %(default)s",
    )
    MultiModalOpenMax_params.add_argument(
        "--distance_metric",
        default="cosine",
        type=str,
        choices=list(pairwisedistances.implemented_distances),
        help="distance metric to use default: %(default)s",
    )
    MultiModalOpenMax_params.add_argument(
        "--Clustering_Algo",
        default="finch",
        type=str,
        choices=["KMeans", "dbscan", "finch"],
        help="Clustering algorithm used for multi modal openmax default: %(default)s",
    )
    MultiModalOpenMax_params.add_argument(
        "--distances_unique",
        action="store_true",
        default=False,
        help="Use unique distances during fitting",
    )
    return parser, dict(
        group_parser=MultiModalOpenMax_params,
        param_names=("Clustering_Algo", "tailsize", "distance_multiplier"),
        param_id_string="{}_TS_{}_DM_{:.2f}",
    )


def MultiModalOpenMax_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the EVM_Params function above.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    """
    from ..clusteringAlgos import clustering

    device = "cpu" if gpu == -1 else f"cuda:{gpu}"

    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name]
        # clustering
        Clustering_Algo = getattr(clustering, args.Clustering_Algo)

        features = features.type(torch.FloatTensor)
        centroids, assignments = Clustering_Algo(
            features,
            K=min(features.shape[0], 100),
            verbose=False,
            distance_metric=args.distance_metric,
        )
        features = features.to(device)
        centroids = centroids.type(features.dtype)
        # TODO: This grid search is not optimized for speed due to redundant distance computation,
        #  needs to be improved if grid search for MultiModal OpenMax is used extensively on big datasets.
        for tailsize, distance_multiplier in itertools.product(
            args.tailsize, args.distance_multiplier
        ):
            MAVs = []
            wbFits = []
            smallScoreTensor = []
            for MAV_no in set(assignments.cpu().tolist()) - {-1}:
                MAV = centroids[MAV_no, :].to(device)
                f = features[assignments == MAV_no].to(device)
                distances = pairwisedistances.__dict__[args.distance_metric](
                    f, MAV[None, :]
                )
                # check if unique distances are desired
                if args.distances_unique:
                    distances = torch.unique(distances)[:, None]

                # Rather than continuing now fit_high handels this by returning invalid weibul shape, scale
                # if distances.shape[0] <= 5:
                #     continue
                weibull_model = fit_high(distances.T, distance_multiplier, tailsize, args.translateAmount)
                MAVs.append(MAV)
                wbFits.append(weibull_model.wbFits)
                smallScoreTensor.append(weibull_model.smallScoreTensor)
            if len(wbFits) == 0:
                yield (
                    f"{args.Clustering_Algo}_TS_{tailsize}_DM_{distance_multiplier:.2f}",
                    (pos_cls_name, None),
                )
                continue
            wbFits = torch.cat(wbFits)
            MAVs = torch.stack(MAVs)
            smallScoreTensor = torch.cat(smallScoreTensor)
            mr = weibull.weibull(
                dict(
                    Scale=wbFits[:, 1],
                    Shape=wbFits[:, 0],
                    signTensor=weibull_model.sign,
                    translateAmountTensor=args.translateAmount,
                    smallScoreTensor=smallScoreTensor,
                )
            )
            mr.tocpu()
            yield (
                f"{args.Clustering_Algo}_TS_{tailsize}_DM_{distance_multiplier:.2f}",
                (pos_cls_name, dict(MAVs=MAVs.cpu(), weibulls=mr)),
            )


def MultiModalOpenMax_Inference(
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
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    for batch_to_process in pos_classes_to_process:
        test_cls_feature = features_all_classes[batch_to_process].to(device)
        probs = []
        for cls_no, cls_name in enumerate(sorted(models.keys())):
            distances = pairwisedistances.__dict__[args.distance_metric](
                test_cls_feature, models[cls_name]["MAVs"].to(device).double()
            )
            probs_current_class = 1 - models[cls_name]["weibulls"].wscore(distances)
            probs.append(torch.max(probs_current_class, dim=1).values)
        probs = torch.stack(probs, dim=-1).cpu()
        yield ("probs", (batch_to_process, probs))
