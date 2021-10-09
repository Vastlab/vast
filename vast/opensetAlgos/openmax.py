import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull


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
        choices=["cosine", "euclidean"],
        help="distance metric to use default: %(default)s",
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
    mr = weibull.weibull()
    mr.FitHigh(distances.double() * distance_multiplier, tailsize, isSorted=False)
    mr.tocpu()
    return mr


def OpenMax_Training(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name].clone().to(f"cuda:{gpu}")
        MAV = torch.mean(features, dim=0).to(f"cuda:{gpu}")
        distances = pairwisedistances.__dict__[args.distance_metric](
            features, MAV[None, :]
        )
        for tailsize, distance_multiplier in itertools.product(
            args.tailsize, args.distance_multiplier
        ):
            weibull_model = fit_high(distances.T, distance_multiplier, tailsize)
            yield (
                f"TS_{tailsize}_DM_{distance_multiplier:.2f}",
                (pos_cls_name, dict(MAV=MAV.cpu()[None, :], weibulls=weibull_model)),
            )


def OpenMax_Inference(pos_classes_to_process, features_all_classes, args, gpu, models):
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
        probs = []
        for class_name in sorted(models.keys()):
            MAV = models[class_name]["MAV"].to(f"cuda:{gpu}")
            distances = pairwisedistances.__dict__[args.distance_metric](features, MAV)
            probs.append(1 - models[class_name]["weibulls"].wscore(distances.cpu()))
        probs = torch.cat(probs, dim=1)
        yield ("probs", (pos_cls_name, probs))
