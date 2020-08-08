import torch
from utile import tools

def tensor_OSRC(gt, predicted_class, score):
    score = tools.device(score)
    score, indices = torch.sort(score,descending=True)
    predicted_class, gt = predicted_class[indices], gt[indices]

    # Reverse score order so that the last occurence of the highest threshold is preserved
    scores_reversed = score[torch.arange(score.shape[0]-1,-1,-1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(scores_reversed,return_counts=True)
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0]-1,-1,-1)
    unique_scores , counts = unique_scores_reversed[indx], counts_reversed[indx]

    # Get the labels for unknowns
    unknown_labels = set(gt.tolist())-set(predicted_class.tolist())

    # Get all indices for knowns and unknowns
    all_known_indexs=[]
    for unknown_label in unknown_labels:
        all_known_indexs.append(gt != unknown_label)
    all_known_indexs = torch.stack(all_known_indexs)
    known_indexs = all_known_indexs.all(dim=0)
    unknown_indexs = ~known_indexs

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum()
    no_of_unknowns = unknown_indexs.sum()

    all_unknowns = torch.cumsum(unknown_indexs,dim=-1).type('torch.FloatTensor')
    OSE = all_unknowns/no_of_unknowns

    correct = gt==predicted_class
    correct = torch.cumsum(correct,dim=-1).type('torch.FloatTensor')
    knowns_accuracy = correct / no_of_knowns

    threshold_indices = torch.cumsum(counts,dim=-1)-1
    knowns_accuracy, OSE = knowns_accuracy[threshold_indices], OSE[threshold_indices]
    return (knowns_accuracy,OSE)
