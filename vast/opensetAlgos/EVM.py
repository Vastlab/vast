"""
Author: Akshay Raj Dhamija

@article{rudd2017extreme,
  title={The extreme value machine},
  author={Rudd, Ethan M and Jain, Lalit P and Scheirer, Walter J and Boult, Terrance E},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={40},
  number={3},
  pages={762--768},
  year={2017},
  publisher={IEEE}
}
"""
import warnings
import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from typing import Iterator, Tuple, List, Dict


def EVM_Params(parser):
    EVM_params_parser = parser.add_argument_group(title="EVM", description="EVM params")
    EVM_params_parser.add_argument(
        "--tailsize",
        nargs="+",
        type=float,
        default=[1.0],
        help="tail size to use\ndefault: %(default)s",
    )
    EVM_params_parser.add_argument(
        "--cover_threshold",
        nargs="+",
        type=float,
        default=[0.7],
        help="cover threshold to use\ndefault: %(default)s",
    )
    EVM_params_parser.add_argument(
        "--distance_multiplier",
        nargs="+",
        type=float,
        default=[0.55],
        help="distance multiplier to use\ndefault: %(default)s",
    )
    EVM_params_parser.add_argument(
        "--distance_metric",
        default="euclidean",
        type=str,
        choices=["cosine", "euclidean"],
        help="distance metric to use\ndefault: %(default)s",
    )
    EVM_params_parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Number of classes per chunk, reduce this parameter if facing OOM "
        "error\ndefault: %(default)s",
    )
    return parser, dict(
        group_parser=EVM_params_parser,
        param_names=("tailsize", "distance_multiplier", "cover_threshold"),
        param_id_string="TS_{}_DM_{:.2f}_CT_{:.2f}",
    )


def fit_low(distances, distance_multiplier, tailsize, gpu):
    mr = weibull.weibull()
    mr.FitLow(
        distances.double() * distance_multiplier,
        min(tailsize, distances.shape[1]),
        isSorted=False,
        gpu=gpu,
    )
    return mr


def set_cover(mr_model, positive_distances, cover_threshold):
    # compute probabilities
    probabilities = mr_model.wscore(positive_distances)

    # threshold by cover threshold
    e = torch.eye(probabilities.shape[0]).type(torch.BoolTensor)
    thresholded = probabilities >= cover_threshold
    thresholded[e] = True
    del probabilities

    # greedily add points that cover most of the others
    covered = torch.zeros(thresholded.shape[0]).type(torch.bool)
    extreme_vectors = []
    covered_vectors = []

    while not torch.all(covered).item():
        sorted_indices = torch.topk(
            torch.sum(thresholded[:, ~covered], dim=1),
            len(extreme_vectors) + 1,
            sorted=False,
        ).indices
        for indx, sortedInd in enumerate(sorted_indices.tolist()):
            if sortedInd not in extreme_vectors:
                break
        else:
            print(thresholded.device, "ENTERING INFINITE LOOP ... EXITING")
            break
        covered_by_current_ev = torch.nonzero(thresholded[sortedInd, :], as_tuple=False)
        covered[covered_by_current_ev] = True
        extreme_vectors.append(sortedInd)
        covered_vectors.append(covered_by_current_ev.to("cpu"))
    del covered
    extreme_vectors_indexes = torch.tensor(extreme_vectors)
    params = mr_model.return_all_parameters()
    scale = torch.gather(params["Scale"].to("cpu"), 0, extreme_vectors_indexes)
    shape = torch.gather(params["Shape"].to("cpu"), 0, extreme_vectors_indexes)
    smallScore = torch.gather(
        params["smallScoreTensor"][:, 0].to("cpu"), 0, extreme_vectors_indexes
    )
    extreme_vectors_models = weibull.weibull(
        dict(
            Scale=scale,
            Shape=shape,
            signTensor=params["signTensor"],
            translateAmountTensor=params["translateAmountTensor"],
            smallScoreTensor=smallScore,
        )
    )
    del params
    return (extreme_vectors_models, extreme_vectors_indexes, covered_vectors)


def EVM_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    :param pos_classes_to_process: List of class names to be processed by this function in the current process class.
    :param features_all_classes: features of all classes, note the classes in pos_classes_to_process can be a subset of the keys for this dictionary
    :param args: A named tuple or an argument parser object containing the arguments mentioned in the EVM_Params function above.
    :param gpu: An integer corresponding to the gpu number to use by the current process.
    :param models: Not used during training, input ignored.
    :return: Iterator(Tuple(parameter combination identifier, Tuple(class name, its evm model)))
    TODO: Currently the training needs gpus, there is no cpu version for now since it is low priority.

    For using with multiprocessing:
    This function has been designed such that computational work load can be easily split into different processes while
    reducing redundant work done for grid search. Each process is only responsible for creating models for classes mentioned
    in the pos_classes_to_process list. For these classes they would create models for each of the possible hyper parameter
    combinations. Please note pos_classes_to_process only contains the names of classes or the keys in features_all_classes
    that need to be processed. The actual features are held in features_all_classes and can be in shared memory across
    different processes. Though the pos_classes_to_process variable changes per process features_all_classes does not.
    For a detailed example that exploits these features please visit https://github.com/akshay-raj-dhamija/vel.


    Grid Search Capabilities:
    This function is also capable of performing grid search for hyper parameters such as tailsize, distance multiplier
    and cover threshold. Which are expected to be passed as list in the args variable, whose elements are supposed to be
    the ones mentioned in the EVM_Params function. The most compute expensive part of EVM is the pairwise distance
    calculation, which is not impacted by the hyper parameters for which grid search is performed.
    Our approach drastically reduces the computation time by re-utilizing the pairwise distance computation across hyper
    parameter combinations.
    TODO: The computation time can be further reduced by reusing weibull fitting for different cover threshold parameters.


    Memory issues:
    It must be noted that the maximum possible tail size being considered for grid search can impact the memory consumption.
    A variable to help reduce the memory consumption is the chunk_size parameter, the lesser the chunk_size, the lesser
    memory is requiered. While this parameter can be very helpful when running the EVM in multiple processes, it might
    not help if the number of classes being handled by the current process is equivalent to the total number of classes.
    TODO: Convert args.chunk_size from number of classes per batch to number of samples per batch. This would be useful
    for handeling highly unbalanced number of samples per class.


    Models provided by this function:
    This function provides models in a partitioned way using an Iterator.
    It would only provide models for the classes mentioned in the pos_classes_to_process.
    At each iterator step it would provide the model for a specific class and a specific hyper parameter combination.
    The results are provided as a Tuple(str, Tuple2), where the str entry tells the hyper parameter combination.
    The Tuple2 contains the name of the class and its corresponding EVM model.
    """
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    negative_classes_for_current_batch = []
    no_of_negative_classes_for_current_batch = 0
    temp = []
    for cls_name in set(features_all_classes.keys()) - set(pos_classes_to_process):
        no_of_negative_classes_for_current_batch += 1
        temp.append(features_all_classes[cls_name])
        if len(temp) == args.chunk_size:
            negative_classes_for_current_batch.append(torch.cat(temp))
            temp = []
    if len(temp) > 0:
        negative_classes_for_current_batch.append(torch.cat(temp))
    for pos_cls_name in pos_classes_to_process:
        # Find positive class features
        positive_cls_feature = features_all_classes[pos_cls_name].to(device)
        tailsize = max(args.tailsize)
        if tailsize <= 1:
            tailsize = tailsize * positive_cls_feature.shape[0]
        tailsize = int(tailsize)

        negative_classes_for_current_class = []
        temp = []
        neg_cls_current_batch = 0
        for cls_name in set(pos_classes_to_process) - {pos_cls_name}:
            neg_cls_current_batch += 1
            temp.append(features_all_classes[cls_name])
            if len(temp) == args.chunk_size:
                negative_classes_for_current_class.append(torch.cat(temp))
                temp = []
        if len(temp) > 0:
            negative_classes_for_current_class.append(torch.cat(temp))
        negative_classes_for_current_class.extend(negative_classes_for_current_batch)

        assert (
            len(negative_classes_for_current_class) >= 1
        ), "In order to train the EVM you need atleast one negative sample for each positive class"
        bottom_k_distances = []
        for batch_no, neg_features in enumerate(negative_classes_for_current_class):
            assert positive_cls_feature.shape[0] != 0 and neg_features.shape[0] != 0, (
                f"Empty tensor encountered positive_cls_feature {positive_cls_feature.shape}"
                f"neg_features {neg_features.shape}"
            )
            distances = pairwisedistances.__dict__[args.distance_metric](
                positive_cls_feature, neg_features.to(device)
            )
            bottom_k_distances.append(distances.cpu())
            bottom_k_distances = torch.cat(bottom_k_distances, dim=1)
            # Store bottom k distances from each batch to the cpu
            bottom_k_distances = [
                torch.topk(
                    bottom_k_distances,
                    min(tailsize, bottom_k_distances.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=True,
                ).values
            ]
            del distances
        bottom_k_distances = bottom_k_distances[0].to(device)

        # Find distances to other samples of same class
        positive_distances = pairwisedistances.__dict__[args.distance_metric](
            positive_cls_feature, positive_cls_feature
        )
        # check if distances to self is zero
        e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
        if not torch.allclose(
            positive_distances[e].type(torch.FloatTensor),
            torch.zeros(positive_distances.shape[0]),
            atol=1e-05,
        ):
            warnings.warn(
                "Distances of samples to themselves is not zero. This may be due to a precision issue or something might be wrong with you distance function."
            )

        for distance_multiplier, cover_threshold, org_tailsize in itertools.product(
            args.distance_multiplier, args.cover_threshold, args.tailsize
        ):
            if org_tailsize <= 1:
                tailsize = int(org_tailsize * positive_cls_feature.shape[0])
            else:
                tailsize = int(org_tailsize)
            # Perform actual EVM training
            weibull_model = fit_low(bottom_k_distances, distance_multiplier, tailsize, gpu)
            extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(
                weibull_model, positive_distances.to(device), cover_threshold
            )
            extreme_vectors = torch.gather(
                positive_cls_feature,
                0,
                extreme_vectors_indexes[:, None]
                .to(device)
                .repeat(1, positive_cls_feature.shape[1]),
            )
            extreme_vectors_models.tocpu()
            extreme_vectors = extreme_vectors.cpu()
            yield (
                f"TS_{org_tailsize}_DM_{distance_multiplier:.2f}_CT_{cover_threshold:.2f}",
                (
                    pos_cls_name,
                    dict(
                        # torch.Tensor -- The extreme vectors used by EVM
                        extreme_vectors=extreme_vectors,
                        # torch.LongTensor -- The index of the above extreme_vectors corresponding to their location in
                        # features_all_classes, only useful if you want to reduce the size of EVM model you save.
                        extreme_vectors_indexes=extreme_vectors_indexes,
                        # weibull.weibull class obj -- the output of weibulls.return_all_parameters() combined with the
                        # extreme_vectors is the actual EVM model for one given class.
                        weibulls=extreme_vectors_models,
                    ),
                ),
            )


def EVM_Inference(
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
        assert test_cls_feature.shape[0] != 0
        probs = []
        for cls_no, cls_name in enumerate(sorted(models.keys())):
            distances = pairwisedistances.__dict__[args.distance_metric](
                test_cls_feature, models[cls_name]["extreme_vectors"].double().to(device)
            )
            probs_current_class = models[cls_name]["weibulls"].wscore(distances)
            probs.append(torch.max(probs_current_class, dim=1).values)
        probs = torch.stack(probs, dim=-1).cpu()
        yield ("probs", (batch_to_process, probs))


class EVM_Inference_cpu_max_knowness_prob:
    """
    This class performs the same function as EVM_Inference but rather than providing per class knowness score, it
    provides maximum knowness score for each sample. It is faster and currently only runs on cpu.
    """

    def __init__(self, distance_metric, models):
        combined_weibull_model = {}
        combined_weibull_model["Scale"] = []
        combined_weibull_model["Shape"] = []
        combined_weibull_model["smallScoreTensor"] = []

        combined_extreme_vectors = []

        for cls_no, cls_name in enumerate(sorted(models.keys())):
            models[cls_name]["weibulls"].tocpu()
            weibull_params_current_cls = models[cls_name][
                "weibulls"
            ].return_all_parameters()
            combined_weibull_model["Scale"].extend(
                weibull_params_current_cls["Scale"].tolist()
            )
            combined_weibull_model["Shape"].extend(
                weibull_params_current_cls["Shape"].tolist()
            )
            combined_weibull_model["smallScoreTensor"].extend(
                weibull_params_current_cls["smallScoreTensor"].tolist()
            )
            combined_extreme_vectors.extend(
                models[cls_name]["extreme_vectors"].cpu().tolist()
            )
        # Only taking the last available values for signTensor and translateAmountTensor
        combined_weibull_model["signTensor"] = weibull_params_current_cls["signTensor"]
        combined_weibull_model["translateAmountTensor"] = weibull_params_current_cls[
            "translateAmountTensor"
        ]
        combined_weibull_model["Scale"] = torch.tensor(combined_weibull_model["Scale"])
        combined_weibull_model["Shape"] = torch.tensor(combined_weibull_model["Shape"])
        combined_weibull_model["smallScoreTensor"] = torch.tensor(
            combined_weibull_model["smallScoreTensor"]
        )
        combined_extreme_vectors = torch.tensor(
            combined_extreme_vectors, dtype=torch.float64
        )

        self.combined_model = {}
        self.combined_model["weibulls"] = weibull.weibull(combined_weibull_model)
        self.combined_model["extreme_vectors"] = combined_extreme_vectors

        self.distance_metric = distance_metric

    def __call__(self, sample_to_process):
        distances = pairwisedistances.__dict__[self.distance_metric](
            sample_to_process[None, :], self.combined_model["extreme_vectors"]
        )
        probs = torch.max(self.combined_model["weibulls"].wscore(distances), dim=1).values
        return probs
