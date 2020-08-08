import numpy as np
import pandas as pd
import torch
from utile import tools

def org_process_each_file(torch_tensor):
    csv_content=pd.DataFrame(torch_tensor.numpy())
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    FP=[0]
    TP=[0]
    N=0
    N_above_UK=0
    UK_prob=1.
    for ind,row in df.iterrows():
        # If Sample is Unknown
        if row[2]==unknowns_label:
            UK_prob=row[4]
            FP.append(FP[-1]+1)
            TP.append(N)
        # If Sample is Known and Classified Correctly
        else:
            if row[1]==row[2]:
                N_above_UK+=1
                if row[4] < UK_prob:
                    N=N_above_UK
                    
    TP=np.array(TP[1:]).astype(np.float32)
    FP=np.array(FP[1:]).astype(np.float32)
    return TP,FP,positives,unknowns

def OSRC(data):
    """
    format for data array
    --------------------------------
    | gt | predicted class | score |
    --------------------------------
    gt is >= 0 for knowns and <0 for unknowns
    """
    no_of_knowns=data[data[:,0]>=0].shape[0]
    no_of_unknowns=data[data[:,0]<0].shape[0]
    knowns=data[data[:,0]>=0].copy()
#    knowns=data[data[:,0]==data[:,1]].copy()
    unknowns=data[data[:,0]<0].copy()
    
    knowns_sorted_array,_,knowns_counts=np.unique(knowns[:,-1],return_index=True,return_counts=True)
    knowns_sorted_array=knowns_sorted_array[::-1]
    knowns_counts=knowns_counts[::-1]
    
    unknowns_sorted_array,_,unknowns_counts=np.unique(unknowns[:,-1],return_index=True,return_counts=True)
    unknowns_sorted_array=unknowns_sorted_array[::-1]
    unknowns_counts=unknowns_counts[::-1]
    
    knowns_array=np.vstack((knowns_counts,np.zeros(knowns_counts.shape[0]),knowns_sorted_array)).transpose()
    unknowns_array=np.vstack((np.zeros(unknowns_counts.shape[0]),unknowns_counts,unknowns_sorted_array)).transpose()
    
    result=np.concatenate((knowns_array, unknowns_array), axis=0)
    sorted_indxs=np.argsort(result[:,-1],axis=0)
    result=result[sorted_indxs[::-1],...]
    print ("result",result)
    TP=np.cumsum(result[:,0])/no_of_knowns
    OSE=np.cumsum(result[:,1])/no_of_unknowns

    return TP,OSE
    """
    fig, ax = plt.subplots()
    for i,(TP,FP,positives,unknowns) in enumerate(to_plot):
        ax.plot(FP/unknowns,TP/positives,label=labels[i])
        u.append(unknowns)
    ax.set_xscale('log')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0,1])
    ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
    ax.set_xlabel('False Positive Rate : Total Unknowns '+str(list(set(u))[0]), fontsize=18, labelpad=10)

    """








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
