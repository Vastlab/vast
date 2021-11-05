import torch

def features_to_dim(feature_dict):
    all_features = []
    for key in feature_dict.keys():
        all_features.append(feature_dict[key])
    
    all_features = torch.cat(all_features, dim=0)
        
    out_feature_dict = {}
    
    for column in range(all_features.shape[1]):
        out_feature_dict[str(column)] = all_features[:, column][:,None].share_memory_()
    
    return out_feature_dict