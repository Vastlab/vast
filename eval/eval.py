import torch
from vast.tools import device,set_device_cpu
set_device_cpu()

def common_processing(gt, predicted_class, score, knownness_score = None):
    """
    Returns samples sorted by scores along with thresholds
    :param gt:
    :param predicted_class:
    :param score:
    :return:
    """
    if knownness_score is None:
        if len(score.shape)!=1:
            knownness_score = score[:,0].clone()
        else:
            knownness_score = score.clone()
    knownness_score = device(knownness_score)

    # Sort samples in decreasing order of knownness
    knownness_score, indices = torch.sort(knownness_score, descending=True)
    indices = indices.cpu()
    predicted_class, gt, score = predicted_class[indices], gt[indices], score[indices]
    del indices

    # Perform score tie breaking
    # The last occurence of the highest threshold is to be preserved

    # sort knownness scores in a ascending order to find unique occurences
    scores_reversed = knownness_score[torch.arange(score.shape[0]-1,-1,-1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(scores_reversed,return_counts=True)
    del scores_reversed
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0]-1,-1,-1)
    unique_scores , counts = unique_scores_reversed[indx], counts_reversed[indx]
    del unique_scores_reversed, counts_reversed

    threshold_indices = torch.cumsum(counts,dim=-1)-1
    return gt, predicted_class, score, unique_scores, threshold_indices

def get_known_unknown_indx(gt, predicted_class):
    # Get the labels for unknowns
    unknown_labels = set(torch.flatten(gt).tolist())-set(torch.flatten(predicted_class).tolist())

    # Get all indices for knowns and unknowns
    all_known_indexs=[]
    for unknown_label in unknown_labels:
        all_known_indexs.append(gt != unknown_label)
    all_known_indexs = torch.stack(all_known_indexs)
    known_indexs = all_known_indexs.all(dim=0)
    unknown_indexs = ~known_indexs
    del all_known_indexs
    return known_indexs, unknown_indexs


def tensor_OSRC(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(gt, predicted_class,
                                                                                     score, knownness_score)
    gt = device(gt)

    known_indexs, unknown_indexs = get_known_unknown_indx(gt, predicted_class)

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum().type('torch.FloatTensor')
    no_of_unknowns = unknown_indexs.sum().type('torch.FloatTensor')

    all_unknowns = torch.cumsum(unknown_indexs,dim=-1).type('torch.FloatTensor')
    OSE = all_unknowns/no_of_unknowns

    correct = torch.any(gt[:,None].cpu()==predicted_class, dim=1)
    correct = device(correct)
    correct = torch.cumsum(correct,dim=-1).type('torch.FloatTensor')

    knowns_accuracy = correct / no_of_knowns
    current_converage = torch.cumsum(known_indexs, dim=0)/no_of_knowns
    knowns_accuracy, current_converage, OSE = knowns_accuracy[threshold_indices],\
                                              current_converage[threshold_indices],\
                                              OSE[threshold_indices]
    return (OSE, knowns_accuracy, current_converage)


def coverage(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(gt, predicted_class,
                                                                                     score, knownness_score)
    correct = torch.any(gt[:,None].cpu()==predicted_class, dim=1)
    correct = torch.cumsum(correct,dim=-1).type('torch.FloatTensor')
    acc = correct[threshold_indices]/gt.shape[0]
    current_converage = (threshold_indices + 1)/gt.shape[0]
    return unique_scores, acc, current_converage
