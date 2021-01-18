import torch
from . import tools

def tensor_OSRC(gt, predicted_class, score):
    if len(score.shape)!=1:
        score = score[:,0]
    score = tools.device(score)
    score, indices = torch.sort(score, descending=True)
    indices = indices.cpu()
    predicted_class, gt = predicted_class[indices], gt[indices]
    del indices

    # Reverse score order so that the last occurence of the highest threshold is preserved
    scores_reversed = score[torch.arange(score.shape[0]-1,-1,-1)]
    unique_scores_reversed, counts_reversed = torch.unique_consecutive(scores_reversed,return_counts=True)
    del scores_reversed
    # Reverse again to get scores & counts in descending order
    indx = torch.arange(unique_scores_reversed.shape[0]-1,-1,-1)
    unique_scores , counts = unique_scores_reversed[indx], counts_reversed[indx]
    del unique_scores_reversed, counts_reversed

    gt = tools.device(gt)
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

    # Get the denominators for accuracy and OSE
    no_of_knowns = known_indexs.sum().type('torch.FloatTensor')
    no_of_unknowns = unknown_indexs.sum().type('torch.FloatTensor')

    all_unknowns = torch.cumsum(unknown_indexs,dim=-1).type('torch.FloatTensor')
    OSE = all_unknowns/no_of_unknowns

    correct = torch.any(gt[:,None].cpu()==predicted_class, dim=1)
    correct = tools.device(correct)
    correct = torch.cumsum(correct,dim=-1).type('torch.FloatTensor')

    knowns_accuracy = correct / no_of_knowns
    threshold_indices = torch.cumsum(counts,dim=-1)-1
    knowns_accuracy, OSE = knowns_accuracy[threshold_indices], OSE[threshold_indices]
    
    return (knowns_accuracy,OSE)
