import itertools
import torch
import torchvision.transforms as transforms
from vast.tools import logger as vastlogger


def ODIN_Params(parser):
    ODIN_Params = parser.add_argument_group("ODIN params")
    ODIN_Params.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
        help="Temperature default: %(default)s",
    )
    ODIN_Params.add_argument(
        "--epsilon",
        nargs="+",
        type=float,
        default=torch.arange(0, 0.004, 0.004 / 10).tolist(),
        help="epsilon default: %(default)s",
    )
    return parser, dict(
        group_parser=ODIN_Params,
        param_names=("temperature", "epsilon"),
        param_id_string="T_{}_E_{}",
    )


def __run_for_all_parameters__(
    args, net, data, normalization, current_class_results, device
):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    softmax_op = torch.nn.Softmax(dim=1)

    data = data.to(device)
    data.requires_grad = True
    original_output = net(data)
    # We want output of the last layer
    if type(original_output) == tuple:
        original_output = original_output[0]
    labels = original_output.detach().argmax(dim=1)

    for temperature in args.temperature:
        # Using temperature scaling
        temperature_scaled_output = original_output / temperature
        loss = criterion(temperature_scaled_output, labels)
        loss.backward(gradient=torch.ones_like(loss), retain_graph=True)

        # Take the sign of the gradient and convert it into actual image max normalized value
        gradient_sign = data.grad.sign().to(device)
        if normalization is not None:
            # Scale gradient values to the actual image normalization values
            gradient_sign = gradient_sign * normalization(
                torch.ones(gradient_sign.shape[1:]).to(device)
            )

        noise_to_add = (
            torch.tensor(args.epsilon)[:, None, None, None].to(device)
            * gradient_sign[0, ...]
        )
        noisy_inputs = noise_to_add + data.detach()
        if normalization is None:
            noisy_inputs = noisy_inputs.clamp(min=0.0, max=1.0)
        noisy_output = net(noisy_inputs)
        if type(noisy_output) == tuple:
            noisy_output = noisy_output[0]
        noisy_output = noisy_output / temperature
        probs = softmax_op(noisy_output)
        scores, prediction = probs.max(dim=1)
        scores, prediction = scores.cpu(), prediction.cpu()

        for epsilon, s, p in zip(args.epsilon, scores.tolist(), prediction.tolist()):
            current_class_results[f"T_{temperature}_E_{epsilon}"].append([s, p])
        data.grad.zero_()
    return current_class_results


def ODIN_Training(dataset, net, args, gpu, state_dict=None):
    """
    ODIN Training only has parameter optimization.
    So the same function can be used for testing as well for a provided set of hyper parameters.
    :param dataset:
    :param net:
    :param args:
    :param gpu:
    :param state_dict:
    :return:
    """
    logger = vastlogger.get_logger()
    # MNIST dataset is not sorted by default, this ensures the sorting.
    # Might need to generalize this to other loader like Image Folder
    targets = torch.tensor(dataset.targets)
    all_classes = sorted(list(set(targets.tolist())))
    quotent, remainder = divmod(len(all_classes), args.world_size)
    classes_to_process = all_classes[
        gpu * quotent + min(gpu, remainder) : (gpu + 1) * quotent + min(gpu + 1, remainder)
    ]
    logger.info(f"Processing classes {classes_to_process}")

    sampler = []
    for cls in classes_to_process:
        sampler.extend(torch.where(targets == cls)[0].tolist())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, sampler=sampler
    )

    if state_dict is not None:
        net.load_state_dict(torch.load(state_dict))
    device = f"cuda:{gpu}"
    net.to(device)
    normalization = None
    if type(dataloader.dataset.transform) == transforms.transforms.Compose:
        for t in dataloader.dataset.transform.transforms:
            if type(t) == transforms.transforms.Normalize:
                normalization = t

    current_class_results = {}
    for temperature, epsilon in itertools.product(args.temperature, args.epsilon):
        current_class_results[f"T_{temperature}_E_{epsilon}"] = []

    current_class = None
    cls_counter = 0
    for data, cls in dataloader:
        cls = cls.item()
        if current_class is None:
            current_class = cls
        if current_class != cls:
            cls_counter += 1
            if cls_counter % 10 == 0:
                logger.info(f"Completed classes {cls_counter}/{len(classes_to_process)}")
            for k in current_class_results:
                yield (k, (current_class, torch.tensor(current_class_results[k])))
                current_class_results[k] = []
            current_class = cls

        current_class_results = __run_for_all_parameters__(
            args, net, data, normalization, current_class_results, device
        )

    for k in current_class_results:
        yield (k, (cls, torch.tensor(current_class_results[k])))
    logger.debug(f"Completed class {cls}")


def ODIN_Inference(dataset, net, args, gpu, state_dict=None):
    return ODIN_Training(dataset, net, args, gpu, state_dict=None)
