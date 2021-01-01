import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull

def EVM_Params(parser):
    EVM_params_parser = parser.add_argument_group(title="EVM",description="EVM params")
    EVM_params_parser.add_argument("--tailsize", nargs="+", type=float, default=[1.0],
                                   help="tail size to use\ndefault: %(default)s")
    EVM_params_parser.add_argument("--cover_threshold", nargs="+", type=float, default=[0.7],
                                   help="cover threshold to use\ndefault: %(default)s")
    EVM_params_parser.add_argument("--distance_multiplier", nargs="+", type=float, default=[0.55],
                                   help="distance multiplier to use\ndefault: %(default)s")
    EVM_params_parser.add_argument('--distance_metric', default='euclidean', type=str, choices=['cosine','euclidean'],
                                   help='distance metric to use\ndefault: %(default)s')
    EVM_params_parser.add_argument("--chunk_size", type=int, default=200,
                                   help="Number of classes per chunk, reduce this parameter if facing OOM "
                                        "error\ndefault: %(default)s")
    return parser, dict(group_parser = EVM_params_parser,
                        param_names = ("tailsize", "distance_multiplier", "cover_threshold"),
                        param_id_string = "TS_{}_DM_{:.2f}_CT_{:.2f}")




def fit_low(distances, distance_multiplier, tailsize, gpu):
    mr = weibull.weibull()
    mr.FitLow(distances.double() * distance_multiplier, min(tailsize,distances.shape[1]), isSorted=False, gpu=gpu)
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
        sorted_indices = torch.topk(torch.sum(thresholded[:, ~covered], dim=1),
                                    len(extreme_vectors)+1,
                                    sorted=False,
                                    ).indices
        for indx, sortedInd in enumerate(sorted_indices.tolist()):
            if sortedInd not in extreme_vectors:
                break
        else:
            print(thresholded.device,"ENTERING INFINITE LOOP ... EXITING")
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
    extreme_vectors_models = weibull.weibull(dict(Scale=scale,
                                                  Shape=shape,
                                                  signTensor=params["signTensor"],
                                                  translateAmountTensor=params["translateAmountTensor"],
                                                  smallScoreTensor=smallScore))
    del params
    return (extreme_vectors_models, extreme_vectors_indexes, covered_vectors)

def EVM_Training(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    # TODO: Convert args.chunk_size from number of classes per batch to number of samples per batch.
    # This would be useful for handeling highly unbalanced number of samples per class.
    negative_classes_for_current_batch = []
    no_of_negative_classes_for_current_batch = 0
    temp = []
    for cls_name in set(features_all_classes.keys()) - set(pos_classes_to_process):
        no_of_negative_classes_for_current_batch+=1
        temp.append(features_all_classes[cls_name])
        if len(temp) == args.chunk_size:
            negative_classes_for_current_batch.append(torch.cat(temp))
            temp = []
    if len(temp) > 0:
        negative_classes_for_current_batch.append(torch.cat(temp))
    for pos_cls_name in pos_classes_to_process:
        # Find positive class features
        positive_cls_feature = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
        tailsize = max(args.tailsize)
        if tailsize<=1:
            tailsize = int(tailsize*positive_cls_feature.shape[0])

        negative_classes_for_current_class=[]
        temp = []
        neg_cls_current_batch=0
        for cls_name in set(pos_classes_to_process)-{pos_cls_name}:
            neg_cls_current_batch+=1
            temp.append(features_all_classes[cls_name])
            if len(temp) == args.chunk_size:
                negative_classes_for_current_class.append(torch.cat(temp))
                temp = []
        if len(temp)>0:
            negative_classes_for_current_class.append(torch.cat(temp))
        negative_classes_for_current_class.extend(negative_classes_for_current_batch)

        bottom_k_distances = []
        negative_distances=[]
        for batch_no, neg_features in enumerate(negative_classes_for_current_class):
            assert positive_cls_feature.shape[0] != 0 and neg_features.shape[0] != 0
            distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature,
                                                                         neg_features.to(f"cuda:{gpu}"))
            bottom_k_distances.append(distances.cpu())
            bottom_k_distances = torch.cat(bottom_k_distances, dim=1)
            # Store bottom k distances from each batch to the cpu
            bottom_k_distances = [torch.topk(bottom_k_distances,
                                             min(tailsize,bottom_k_distances.shape[1]),
                                             dim = 1,
                                             largest = False,
                                             sorted = True).values]
            del distances
        bottom_k_distances = bottom_k_distances[0].to(f"cuda:{gpu}")

        # Find distances to other samples of same class
        positive_distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature, positive_cls_feature)
        # check if distances to self is zero
        e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
        assert torch.allclose(positive_distances[e].type(torch.FloatTensor), \
                              torch.zeros(positive_distances.shape[0]),atol=1e-06) == True, \
            "Distances of samples to themselves is not zero"

        for distance_multiplier, cover_threshold, org_tailsize in itertools.product(args.distance_multiplier,
                                                                                args.cover_threshold, args.tailsize):
            if org_tailsize <= 1:
                tailsize = int(org_tailsize * positive_cls_feature.shape[0])
            else:
                tailsize = org_tailsize
            # Perform actual EVM training
            weibull_model = fit_low(bottom_k_distances, distance_multiplier, tailsize, gpu)
            extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                         positive_distances.to(f"cuda:{gpu}"),
                                                                                         cover_threshold)
            extreme_vectors = torch.gather(positive_cls_feature, 0,
                                           extreme_vectors_indexes[:,None].to(f"cuda:{gpu}").repeat(1,positive_cls_feature.shape[1]))
            extreme_vectors_models.tocpu()
            extreme_vectors = extreme_vectors.cpu()
            yield (f"TS_{org_tailsize}_DM_{distance_multiplier:.2f}_CT_{cover_threshold:.2f}",
                   (pos_cls_name,dict(extreme_vectors = extreme_vectors,
                                      extreme_vectors_indexes=extreme_vectors_indexes,
                                      weibulls = extreme_vectors_models)))


def EVM_Inference(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    for pos_cls_name in pos_classes_to_process:
        test_cls_feature = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
        assert test_cls_feature.shape[0]!=0
        probs=[]
        for cls_no, cls_name in enumerate(sorted(models.keys())):
            distances = pairwisedistances.__dict__[args.distance_metric](test_cls_feature,
                                                                         models[cls_name]['extreme_vectors'].to(f"cuda:{gpu}"))
            probs_current_class = models[cls_name]['weibulls'].wscore(distances)
            probs.append(torch.max(probs_current_class, dim=1).values)
        probs = torch.stack(probs,dim=-1).cpu()
        yield ("probs", (pos_cls_name, probs))
