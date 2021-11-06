import torch
import numpy as np

def features_to_dim(feature_dict, classes_to_process=None):
    if classes_to_process is None:
        classes_to_process = feature_dict.keys()
    all_features = []
    all_class_identifiers = []
    for key in classes_to_process:
        all_features.append(feature_dict[key])
        all_class_identifiers.extend([key]*feature_dict[key].shape[0])
    
    all_features = torch.cat(all_features, dim=0)
    all_class_identifiers = np.array(all_class_identifiers)

    out_feature_dict = {}

    for column in range(all_features.shape[1]):
        out_feature_dict[str(column)] = all_features[:, column][:,None]
    
    return out_feature_dict, all_class_identifiers