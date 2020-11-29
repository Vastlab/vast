import torch
from ..tools import pairwisedistances
from ..DistributionModels import weibull

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

def EVM(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    chunk_size = 200
    negative_classes_for_current_batch = []
    no_of_negative_classes_for_current_batch = 0
    temp = []
    for cls_name in set(features_all_classes.keys()) - set(pos_classes_to_process):
        no_of_negative_classes_for_current_batch+=1
        temp.append(features_all_classes[cls_name])
        if len(temp) == chunk_size:
            negative_classes_for_current_batch.append(torch.cat(temp))
            temp = []
    if len(temp) > 0:
        negative_classes_for_current_batch.append(torch.cat(temp))

    for pos_cls_name in pos_classes_to_process:
        # Find positive class features
        positive_cls_feature = features_all_classes[pos_cls_name].cuda()
        if args.tailsize<=1:
            tailsize = args.tailsize*positive_cls_feature.shape[0]
        else:
            tailsize = args.tailsize
        tailsize = int(tailsize)

        negative_classes_for_current_class=[]
        temp = []
        neg_cls_current_batch=0
        for cls_name in set(pos_classes_to_process)-{pos_cls_name}:
            neg_cls_current_batch+=1
            temp.append(features_all_classes[cls_name])
            if len(temp) == chunk_size:
                negative_classes_for_current_class.append(torch.cat(temp))
                temp = []
        if len(temp)>0:
            negative_classes_for_current_class.append(torch.cat(temp))
        negative_classes_for_current_class.extend(negative_classes_for_current_batch)
        print(f"Negative classes for current class: {no_of_negative_classes_for_current_batch + neg_cls_current_batch}")

        negative_distances=[]
        for batch_no, neg_features in enumerate(negative_classes_for_current_class):
            distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature, neg_features.cuda())
            # Store bottom k distances from each batch to the cpu
            sortedTensor = torch.topk(distances,
                                      min(tailsize,distances.shape[1]),
                                      dim = 1,
                                      largest = False,
                                      sorted = True).values
            del distances
            negative_distances.append(sortedTensor.cpu())

        # Find distances to other samples of same class
        positive_distances = pairwisedistances.__dict__[args.distance_metric](positive_cls_feature, positive_cls_feature)
        # check if distances to self is zero
        e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
        assert torch.allclose(positive_distances[e].type(torch.FloatTensor), \
                              torch.zeros(positive_distances.shape[0]),atol=1e-03) == True, \
            "Distances of samples to themselves is not zero"
        sortedTensor = torch.cat(negative_distances, dim=1).to(f"cuda:{gpu}")

        # Perform actual EVM training
        try:
            weibull_model = fit_low(sortedTensor, args.distance_multiplier, tailsize, gpu)
        except:
            # TODO: RAISE ERROR
            print("Failed a probable reason is you did not give any negative samples to find distances to")
        extreme_vectors_models, extreme_vectors_indexes, covered_vectors = set_cover(weibull_model,
                                                                                     positive_distances.cuda(),
                                                                                     args.cover_threshold)
        extreme_vectors = torch.gather(positive_cls_feature, 0, extreme_vectors_indexes[:,None].cuda().repeat(1,positive_cls_feature.shape[1]))
        extreme_vectors_models.tocpu()
        extreme_vectors = extreme_vectors.cpu()

        yield (pos_cls_name, dict(extreme_vectors = extreme_vectors,
                                  weibulls = extreme_vectors_models))
    print(f"Last Extreme vector shape was {extreme_vectors.shape}")


def EVM_Inference(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    for pos_cls_name in pos_classes_to_process:
        test_cls_feature = features_all_classes[pos_cls_name].cuda()
        probs=[]
        for cls_no, cls_name in enumerate(sorted(models.keys())):
            distances = pairwisedistances.__dict__[args.distance_metric](test_cls_feature,
                                                                         models[cls_name]['extreme_vectors'].cuda())
            probs_current_class = models[cls_name]['weibulls'].wscore(distances)
            probs.append(torch.max(probs_current_class, dim=1).values)
        probs = torch.stack(probs,dim=-1).cpu()
        yield (pos_cls_name,probs)