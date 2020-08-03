import torch
from torch.nn import functional as F
import libmr

class openmax:
    """
    An instance of this class should be created by each process in order to run an optimize version.
    For best results EVT fit for each class should be performed in a separate process by running the fit_per_class_evt function.
    For inference the processes should independently process different samples
    """
    def __init__(self, tail_size=100, evt_models_file=None, load=False,
                 convert_logits_to_softmax=True, unknowns_label=[-1], locks=None):
        self.tail_size=tail_size
        self.convert_logits_to_softmax = convert_logits_to_softmax
        self.locks = locks
        self.unknowns_label = set(unknowns_label)
        self.evt_models_file = evt_models_file
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2)
        if load:
            print("evt_models_file",evt_models_file)
            self.evt_models = torch.load(evt_models_file)
        # else:
        #     print("You need to fit the evt model and later set")

    def save_all_models(self,evt_models):
        if isinstance(evt_models,list):
            to_save = {}
            for cls_no,cls_evt_dict_info in evt_models:
                to_save[cls_no] = cls_evt_dict_info
            evt_models = to_save
        torch.save(evt_models, open(self.evt_models_file, "wb"))

    def fit_per_class_evt(self,features, gpu=None):
        if gpu is not None:
            torch.cuda.set_device(gpu)
        # self.locks.acquire()
        features = torch.tensor(features).cuda()
        MAV = features.mean(dim=0)
        euclidean_distances = self.euclidean_dist_obj(features, MAV.expand_as(features))
        euclidean_distances = euclidean_distances.cpu().tolist()
        MAV = MAV.cpu().tolist()
        torch.cuda.empty_cache()
        # self.locks.release()
        mr = libmr.MR()
        mr.fit_high(euclidean_distances, self.tail_size)
        return dict(model = mr,
                    MAV = MAV)


    def inferencing(self, features, logits, gpu=0):
        torch.cuda.set_device(gpu)
        features = torch.tensor(features).cuda()
        distances = []
        for cls_no in list(set(self.evt_models.keys())-self.unknowns_label):
            MAV = torch.tensor(self.evt_models[cls_no]['MAV']).cuda()
            distances.append((cls_no,self.euclidean_dist_obj(features, MAV.expand_as(features)).cpu().tolist()))
        torch.cuda.empty_cache()
        knowness_scores = []
        for cls_no,cls_distances in distances:
            knowness_scores.append([self.evt_models[cls_no]['model'].w_score(distance) for distance in cls_distances])
        knowness_scores = torch.tensor(knowness_scores).transpose(0,1).cuda()
        logits = torch.tensor(logits).cuda()
        if self.convert_logits_to_softmax:
            prob_scores = F.softmax(logits,dim=1)
        else:
            prob_scores = logits
        prob_scores = prob_scores*knowness_scores
        prob_scores = prob_scores.cpu().detach()
        torch.cuda.empty_cache()
        return prob_scores
