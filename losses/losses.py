import torch
from torch.nn import functional as F

"""
For usage consult https://github.com/Vastlab/MNIST_Experiments/blob/master/MNIST_SoftMax_Training.py
"""

class center_loss:
    def __init__(self, beta=0.1, classes=range(10), fc_layer_dimension=2):
        """
        This class implements center loss introduced in https://ydwen.github.io/papers/WenECCV16.pdf
        :param beta:  The factor by which centers should be updated
        :param classes: A list containing class labels for which we will be computing center loss \
                        Note-> This list should only contain positive numbers, negatives are reserved for unknowns
        :param fc_layer_dimension: The dimension of the layer in which center loss is being computed
        """
        self.beta = beta
        self.centers = dict(zip(classes, torch.zeros((len(classes), fc_layer_dimension)).requires_grad_(requires_grad=False)))
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2)

    def update_centers(self, FV, true_label):
        # find all samples of knowns
        for cls_no in set(true_label[true_label >= 0].tolist()):
            # Equation (4) from the paper
            delta_c = FV[true_label == cls_no].cpu().detach() - self.centers[cls_no]
            delta_c = torch.sum(delta_c, dim=0) / FV[true_label == cls_no].shape[0]
            # Step 6 from Algorithm 1
            self.centers[cls_no] = self.centers[cls_no] + (self.beta * delta_c)

    def __call__(self, FV, true_label):
        # Equation (2) from paper
        loss = torch.zeros(FV.shape[0]).to(FV.device)
        for cls_no in set(true_label.tolist()):
            loss[true_label == cls_no] = self.euclidean_dist_obj(FV[true_label == cls_no],
                                                                 self.centers[cls_no].expand_as(FV[true_label == cls_no]).to(FV.device))
        return torch.mean(loss)


class tensor_center_loss:
    def __init__(self, beta=0.1, classes=range(10), fc_layer_dimension=2, initial_value=None):
        """
        This class implements center loss introduced in https://ydwen.github.io/papers/WenECCV16.pdf
        :param beta:  The factor by which centers should be updated
        :param classes: A list containing class labels for which we will be computing center loss \
                        Note-> This list should only contain positive numbers, negatives are reserved for unknowns
        :param fc_layer_dimension: The dimension of the layer in which center loss is being computed
        """
        self.beta = beta
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2)
        self.centers = torch.zeros((len(classes), fc_layer_dimension)).requires_grad_(requires_grad=False).cuda()
        if initial_value is not None:
            for cls_no in classes:
                self.centers[cls_no] = torch.tensor(initial_value[cls_no])

    def update_centers(self, FV, true_label):
        FV = FV.detach()
        deltas = FV - self.centers[true_label,:].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
#            print(self.centers[cls_no].shape,deltas[true_label==cls_no].shape,torch.mean(self.beta * deltas[true_label==cls_no],dim=0).shape)
            self.centers[cls_no] += (self.beta * torch.mean(deltas[true_label==cls_no],dim=0))

    def __call__(self, FV, true_label):
        # Equation (2) from paper
        loss = self.euclidean_dist_obj(FV,self.centers[true_label,:].to(FV.device))
        return torch.mean(loss)



class objecto_center_loss:
    def __init__(self, alpha=0.1, classes=range(10), fc_layer_dimension=2, ring_size=50):
        """
        :param alpha:  The factor by which learning rate should be updated
        :param classes: A list containing class labels for which we will be computing center loss \
                        Note-> This list should only contain positive numbers, negatives are reserved for unknowns
        :param fc_layer_dimension: The dimension of the layer in which center loss is being computed
        """
        self.alpha = alpha
        self.ring_size = ring_size
        self.centers = dict(zip(classes,
                                torch.zeros((len(classes), fc_layer_dimension)).requires_grad_(requires_grad=False)
                            ))
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2)

    def update_centers(self, fc_outputs, true_label):
        # find all samples of knowns
        for cls_no in set(true_label[true_label >= 0].tolist()):
            # Equation (4) from the paper
            delta_c = fc_outputs[true_label == cls_no].cpu().detach() - self.centers[cls_no]
            delta_c = torch.sum(delta_c, dim=0) / fc_outputs[true_label == cls_no].shape[0]

            delta_c = delta_c/torch.norm(delta_c, p=2, dim=-1)
            # Update Direction
            self.centers[cls_no] = self.centers[cls_no] + (self.alpha * delta_c)
            self.centers[cls_no] = self.ring_size*(self.centers[cls_no]/torch.norm(self.centers[cls_no], p=2, dim=-1))

    def __call__(self, fc, true_label):
        # Equation (2) from paper
        loss = torch.zeros(fc.shape[0]).to(fc.device)
        for cls_no in set(true_label.tolist()):
            loss[true_label == cls_no] = self.euclidean_dist_obj(fc[true_label == cls_no],
                                                              self.centers[cls_no].expand_as(fc[true_label == cls_no]).to(
                                                                  fc.device))
        return torch.mean(loss)

def entropic_openset_loss(logit_values, target, num_of_classes=10, sample_weights=None):
    catagorical_targets = torch.zeros(logit_values.shape)
    catagorical_targets[target != -1, :] = torch.eye(num_of_classes)[target[target != -1]]
    catagorical_targets[target == -1, :] = torch.ones(target[target == -1].shape[0], num_of_classes) * (
            1. / num_of_classes)
    catagorical_targets = catagorical_targets.to(logit_values.device)
    log_values = F.log_softmax(logit_values, dim=1)
    negative_log_values = -1 * log_values
    loss = negative_log_values * catagorical_targets
    sample_loss = torch.mean(loss, dim=1)
    if sample_weights is not None:
        sample_loss = sample_loss * sample_weights
    return sample_loss


def objectoSphere_loss(features, target, knownsMinimumMag=50., sample_weights=None):
    knownsMinimumMag_tensor = torch.ones((features.shape[0])) * knownsMinimumMag
    knownsMinimumMag_tensor = knownsMinimumMag_tensor.to(features.device)
    mag = features.norm(p=2, dim=1)
    # For knowns magnitude minus \beta is loss
    mag_diff_from_ring = torch.clamp(knownsMinimumMag_tensor - mag, min=0.)
    loss = torch.zeros((features.shape[0])).to(features.device)
    loss[target != -1] = mag_diff_from_ring[target != -1]
    loss[target == -1] = mag[target == -1]
    loss = torch.pow(loss, 2)
    if sample_weights is not None:
        loss = sample_weights * loss
    return loss


def nll_loss(logit_values, target):
    log_values = F.log_softmax(logit_values, dim=1)
    negative_log_values = -1 * log_values
    loss = negative_log_values * target
    sample_loss = torch.mean(loss, dim=1)
    return sample_loss

def org_sigmoid(logit_values, target, sample_weights):
    """
    Reimplementation of original sigmoid loss
    """
    loss = torch._C._nn.binary_cross_entropy(torch.sigmoid(logit_values), target)
    loss = torch.mean(loss*sample_weights,dim=1)
    return torch.mean(loss)
