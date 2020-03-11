import torch

'''
use "update_centers(.)" after optimizer.step() in dataloader loop

'''


class center_loss:
    def __init__(self, alpha=0.1, classes=range(10), fc_layer_dimension=2):
        """
        This class implements center loss introduced in https://ydwen.github.io/papers/WenECCV16.pdf
        :param alpha:  The factor by which learning rate should be updated
        :param classes: A list containing class labels for which we will be computing center loss \
                        Note-> This list should only contain positive numbers, negatives are reserved for unknowns
        :param fc_layer_dimension: The dimension of the layer in which center loss is being computed
        """
        self.alpha = alpha
        self.centers = dict(zip(classes,
                                torch.zeros((len(classes), fc_layer_dimension)).requires_grad_(requires_grad=False)
                                ))
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2)

    def update_centers(self, fc, true_label):
        # find all samples of knowns
        for cls_no in set(true_label[true_label >= 0].tolist()):
            # Equation (4) from the paper
            delta_c = fc[true_label == cls_no].cpu().detach() - self.centers[cls_no]
            delta_c = torch.sum(delta_c, dim=0) / fc[true_label == cls_no].shape[0]
            # Step 6 from Algorithm 1
            self.centers[cls_no] = self.centers[cls_no] + (self.alpha * delta_c)

    def compute_loss(self, fc, true_label):
        # Equation (2) from paper
        loss = torch.zeros(fc.shape[0]).to(fc.device)
        for cls_no in set(true_label.tolist()):
            loss[true_label == cls_no] = self.euclidean_dist_obj(fc[true_label == cls_no],
                                                                 self.centers[cls_no].expand_as(
                                                                     fc[true_label == cls_no]).to(
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
    return torch.mean(loss)
