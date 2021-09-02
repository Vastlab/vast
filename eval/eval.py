import torch
from vast.tools import device,set_device_cpu
set_device_cpu()

def common_processing(gt, predicted_class, score, knownness_score = None):
    """
    Returns samples sorted by knownness scores along with unique scores/thresholds
    :param gt:
    :param predicted_class:
    :param score:
    :return:
    """
    if knownness_score is None:
        if len(score.shape)!=1:
            knownness_score = torch.max(score, dim=1).values
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

    # sort knownness scores in an ascending order to find unique occurences
    scores_reversed = knownness_score[torch.arange(knownness_score.shape[0]-1,-1,-1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(scores_reversed,return_counts=True)
    del scores_reversed
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0]-1,-1,-1)
    unique_scores , counts = unique_scores_reversed[indx], counts_reversed[indx]
    del unique_scores_reversed, counts_reversed

    threshold_indices = torch.cumsum(counts,dim=-1)-1
    return gt, predicted_class, score, unique_scores, threshold_indices

def get_known_unknown_indx(gt, predicted_class, unknown_labels = {-1}):
    # Get all indices for knowns and unknowns
    all_known_indexs=[]
    for unknown_label in unknown_labels:
        all_known_indexs.append(gt != unknown_label)
    all_known_indexs = torch.stack(all_known_indexs)
    known_indexs = all_known_indexs.all(dim=0)
    unknown_indexs = ~known_indexs
    del all_known_indexs
    return known_indexs, unknown_indexs

def get_correct_for_accuracy(gt, predicted_class, score, topk=1):
    score, prediction_indx = torch.topk(score, topk, dim=1)
    prediction_made = torch.gather(predicted_class, 1, prediction_indx)
    correct_bool = torch.any(gt[:,None].cpu()==prediction_made, dim=1)
    correct_cumsum = torch.cumsum(correct_bool,dim=-1).type('torch.FloatTensor')
    return correct_bool, correct_cumsum

def tensor_OSRC(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(gt, predicted_class,
                                                                                     score, knownness_score)
    gt = device(gt)

    known_indexs, unknown_indexs = get_known_unknown_indx(gt, predicted_class)

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum().type('torch.FloatTensor')
    no_of_unknowns = unknown_indexs.sum().type('torch.FloatTensor')

    # any known samples seen for a score > \theta is a known unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    knowns_not_detected_as_unknowns = known_indexs
    knowns_not_detected_as_unknowns[torch.any(predicted_class==-1, dim=1)]=False
    current_converage = torch.cumsum(knowns_not_detected_as_unknowns, dim=0)/no_of_knowns

    correct_bool, _ = get_correct_for_accuracy(gt, predicted_class, score)
    # if a known was detected as unknown it should not be marked as correct
    correct_bool = correct_bool * knowns_not_detected_as_unknowns
    correct_cumsum = torch.cumsum(correct_bool, dim=-1).type('torch.FloatTensor')
    knowns_accuracy = correct_cumsum / no_of_knowns

    # any unknown sample seen for a score > \theta is a false positive unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    unknown_detected_as_known = unknown_indexs
    unknown_detected_as_known[torch.any(predicted_class==-1, dim=1)]=False
    all_FPs = torch.cumsum(unknown_detected_as_known,dim=-1).type('torch.FloatTensor')
    OSE = all_FPs/no_of_unknowns

    knowns_accuracy, current_converage, OSE = knowns_accuracy[threshold_indices],\
                                              current_converage[threshold_indices],\
                                              OSE[threshold_indices]
    return (OSE, knowns_accuracy, current_converage)


def coverage(gt, predicted_class, score, knownness_score=None):
    gt, predicted_class, score, unique_scores, threshold_indices = common_processing(gt, predicted_class,
                                                                                     score, knownness_score)
    correct_bool, correct_cumsum = get_correct_for_accuracy(gt, predicted_class, score)
    acc = correct_cumsum[threshold_indices]/gt.shape[0]
    incorrect_cumsum = torch.cumsum(~correct_bool, dim=-1).type('torch.FloatTensor')
    incorrect_cumsum = incorrect_cumsum[threshold_indices]
    current_converage = (threshold_indices + 1)/gt.shape[0]
    return unique_scores, incorrect_cumsum, acc, current_converage

def calculate_binary_precision_recall(gt, predicted_class, score):
    """
                                Detected as
                            Knowns        Unknowns
    Ground Truth  Knowns      TP             FN
                Unknowns      FP             TN
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

    # any known samples seen for a score > \theta is a known unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    knowns_not_detected_as_unknowns = known_indexs
    knowns_not_detected_as_unknowns[torch.any(predicted_class==-1, dim=1)]=False
    all_knowns = torch.cumsum(knowns_not_detected_as_unknowns, dim=-1).type('torch.FloatTensor')


    # any unknown sample seen for a score > \theta is a false positive unless the sample was predicted as an unknown, i.e. predicted_class == -1 \
    # Note: incase of topk if any of the top k predictions is -1 then it is considered unknown
    unknown_detected_as_known = unknown_indexs
    unknown_detected_as_known[torch.any(predicted_class==-1, dim=1)]=False
    all_unknowns = torch.cumsum(unknown_detected_as_known, dim=-1).type('torch.FloatTensor')

    all_knowns, all_unknowns = all_knowns[threshold_indices], all_unknowns[threshold_indices]

    Recall = all_knowns / no_of_knowns
    # Precision here is non monotonic
    Precision = all_knowns / (all_knowns + all_unknowns)

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
    FScore = (1+(ß**2))*((Precision * Recall) / (((ß**2)*Precision) + Recall))
    return FScore


