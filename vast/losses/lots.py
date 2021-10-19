import torch


def lots(
    network,
    data,
    target,
    target_class=None,
    epsilon=None,
    stepwidth=1.0 / 255.0,
    iterations=1,
):
    """Computes LOTS adversarial for the given input (which should be a single image)
    and returns the adversarial image and whether it was classified as the intended target_class or has reached the desired epsilon.

    :param network: The network that extracts the logits and the deep features
    :param data: The input image
    :param target: The target that should be reached (this is in deep feature dimensions)
    :param target_class: The target class as int. If given, LOTS will stop as soon as the target class is reached.
    :param epsilon: The distance between target and deep feature representation of the adversarial. If given, LOTS will stop when this distance is reached.
    :param stepwidth: The size of one step of LOTS. Can be small for iterative approach, and large for single step.
    :param iterations: The maximum number of iterations that LOTS should run.

    :return adversarial: The adversarial image that has been created
    :return reached_target: Boolean indicating whether the target has been reached.
    """
    # add required batch dimension
    target = torch.unsqueeze(target, 0)
    data = data.unsqueeze(0)
    # iterate for the given number of iterations
    for iteration in range(iterations):
        data.requires_grad_(True)
        network.zero_grad()

        # forward image and extract the given layer and the logits
        logits, features = network.forward(data)

        with torch.no_grad():
            # check if already correctly classified
            if target_class is not None:
                if torch.max(logits, dim=-1)[1] == target_class:
                    return data.squeeze(0), True

            # check if we are close enough to our target
            # this is not used in our small example but only given as possible success criterion
            if epsilon is not None:
                if torch.norm(features - target) < epsilon:
                    return data.squeeze(0), True

        # compute MSE loss between output and target
        loss = torch.nn.functional.mse_loss(features, target, reduction="sum")

        # get gradient
        loss.backward()
        gradient = data.grad.detach()
        with torch.no_grad():
            # normalize gradient to have its max abs value at 1
            gradient_step = gradient * (stepwidth / torch.max(torch.abs(gradient)))

            # reduce loss by moving toward negative gradient, and assure to be in image dimensions
            data = torch.clamp(data - gradient_step, 0.0, 1.0)

    # target has not been reached in the given number of iterations
    return data.squeeze(0), False


def lots_(network, data, target, stepwidth):
    """Computes single-step LOTS adversarial for the given input (which should be a batch of images)
    and returns the adversarial images.

    :param network: The network that extracts the logits and the deep features
    :param data: The input images as a full batch
    :param target: The targets that should be reached (this is in deep feature dimensions), a full batch of targets is required
    :param stepwidth: The size of the step of LOTS.

    :return adversarial: The adversarial image that has been created
    """
    data.requires_grad_(True)
    network.zero_grad()

    # forward image and extract the given layer and the logits
    logits, features = network.forward(data)

    # compute MSE loss between output and target
    loss = torch.nn.functional.mse_loss(features, target, reduction="mean")

    # get gradient
    loss.backward()
    gradient = data.grad.detach()
    with torch.no_grad():
        # normalize gradient to have its max abs value at 1
        N = gradient.size(0)
        norm = torch.max(torch.abs(gradient.view(N, -1)), dim=-1)[0].reshape((N, 1, 1, 1))
        gradient_step = gradient * (stepwidth / norm)

        # reduce loss by moving toward negative gradient, and assure to be in image dimensions
        data = torch.clamp(data - gradient_step, 0.0, 1.0)

    return data
