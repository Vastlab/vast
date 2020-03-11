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
                                                              self.centers[cls_no].expand_as(fc[true_label == cls_no]).to(
                                                                  fc.device))
        return torch.mean(loss)
