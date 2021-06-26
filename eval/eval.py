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


def calculate_binary_precision_recall(gt, predicted_class, score):
    """
    This function measures the performance of an algorithm to identify a known as a known while
    the unknowns only impact as false positives in the precision
    :param gt:
    :param predicted_class:
    :param score:
    :return:
    """
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(gt, predicted_class, score)

    known_indexs, unknown_indexs = get_known_unknown_indx(gt, predicted_class)

    no_of_knowns = known_indexs.sum().type('torch.FloatTensor')
    no_of_unknowns = unknown_indexs.sum().type('torch.FloatTensor')

    all_knowns = torch.cumsum(known_indexs,dim=-1).type('torch.FloatTensor')
    all_unknowns = torch.cumsum(unknown_indexs,dim=-1).type('torch.FloatTensor')
    all_knowns, all_unknowns = all_knowns[threshold_indices], all_unknowns[threshold_indices]

    Recall = all_knowns / no_of_knowns
    # Precision here is non monotonic
    Precision = all_knowns / (all_knowns + all_unknowns)

    Recall = [0.] + Recall.tolist() + [1.]
    Precision = [0.] + Precision.tolist() + [0.]

    # make precision monotonic
    for index_ in range(len(Precision) - 1, 0, -1):
        Precision[index_ - 1] = max(Precision[index_ - 1], Precision[index_])

    Recall = torch.tensor(Recall)
    Precision = torch.tensor(Precision)
    unique_scores = [torch.max(unique_scores).item()] + unique_scores.tolist() + [torch.min(unique_scores).item()]

    return Precision, Recall, unique_scores

def F_score(Precision, Recall, ß=1.):
    """
    Calculates F Score by the following equation, default is F1 score because ß = 1.0
    F_Score = (1+ß**2)*((Precision * Recall) / ((ß**2)*Precision + Recall))
    :param Precision:
    :param Recall:
    :param ß:
    :return:
    """
    FScore = (1+ß**2)*((Precision * Recall) / ((ß**2)*Precision + Recall))
    return FScore