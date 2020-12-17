import torch
import itertools
from ..tools import pairwisedistances
from ..DistributionModels import weibull
from .openmax import fit_high

def MultiModalOpenMax_Params(parser):
    MultiModalOpenMax_params = parser.add_argument_group('MultiModalOpenMax params')
    MultiModalOpenMax_params.add_argument("--tailsize", nargs="+", type=float, default=[1.0],
                                          help="tail size to use default: %(default)s")
    MultiModalOpenMax_params.add_argument("--distance_multiplier", nargs="+", type=float, default=[1.0],
                                          help="distance multiplier to use default: %(default)s")
    MultiModalOpenMax_params.add_argument('--distance_metric', default='cosine', type=str, choices=['cosine','euclidean'],
                                          help='distance metric to use default: %(default)s')
    MultiModalOpenMax_params.add_argument('--Clustering_Algo', default='finch', type=str,
                                          choices=['KMeans','dbscan','finch'],
                                          help='Clustering algorithm used for multi modal openmax default: %(default)s')
    return parser

def MultiModalOpenMax_Training(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    from ..clusteringAlgos import clustering
    for pos_cls_name in pos_classes_to_process:
        features = features_all_classes[pos_cls_name]
        # clustering
        Clustering_Algo = getattr(clustering, args.Clustering_Algo)

        features = features.type(torch.FloatTensor)
        centroids, assignments = Clustering_Algo(features, K=min(features.shape[0], 100), verbose=False,
                                                 distance_metric=args.distance_metric)
        features = features.cuda()
        centroids = centroids.type(features.dtype)
        # TODO: This grid search is not optimized for speed due to redundant distance computation,
        #  needs to be improved if grid search for MultiModal OpenMax is used extensively on big datasets.
        for tailsize, distance_multiplier in itertools.product(args.tailsize, args.distance_multiplier):
            MAVs=[]
            wbFits=[]
            smallScoreTensor=[]
            for MAV_no in set(assignments.cpu().tolist())-{-1}:
                MAV = centroids[MAV_no,:].cuda()
                f = features[assignments == MAV_no].cuda()
                distances = pairwisedistances.__dict__[args.distance_metric](f, MAV[None,:])
                weibull_model = fit_high(distances.T, distance_multiplier, tailsize)
                MAVs.append(MAV)
                wbFits.append(weibull_model.wbFits)
                smallScoreTensor.append(weibull_model.smallScoreTensor)
            wbFits=torch.cat(wbFits)
            MAVs=torch.stack(MAVs)
            smallScoreTensor=torch.cat(smallScoreTensor)
            mr = weibull.weibull(dict(Scale=wbFits[:,1],
                                      Shape=wbFits[:,0],
                                      signTensor=weibull_model.sign,
                                      translateAmountTensor=None,
                                      smallScoreTensor=smallScoreTensor))
            mr.tocpu()
            yield (f"TS_{tailsize}_DM_{distance_multiplier:.2f}",
                   (pos_cls_name, dict(MAVs = MAVs.cpu(),
                                       weibulls = mr)))

def MultiModalOpenMax_Inference(pos_classes_to_process, features_all_classes, args, gpu, models=None):
    for pos_cls_name in pos_classes_to_process:
        test_cls_feature = features_all_classes[pos_cls_name].to(f"cuda:{gpu}")
        probs=[]
        for cls_no, cls_name in enumerate(sorted(models.keys())):
            distances = pairwisedistances.__dict__[args.distance_metric](test_cls_feature,
                                                                         models[cls_name]['MAVs'].cuda().double())
            probs_current_class = 1-models[cls_name]['weibulls'].wscore(distances)
            probs.append(torch.max(probs_current_class, dim=1).values)
        probs = torch.stack(probs,dim=-1).cpu()
        yield ("probs", (pos_cls_name, probs))
