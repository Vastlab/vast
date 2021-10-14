import argparse
import os
import numpy as np
import h5py
import pathlib
import itertools
import multiprocessing as mp
from tqdm import tqdm
import torch
import torchvision

try:
    # https://github.com/pytorch/accimage
    torchvision.set_image_backend("accimage")
except:
    pass
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
from vast.tools import logger as vastlogger

try:
    from pl_bolts.models import self_supervised

    pl_bolts = True
except:
    pl_bolts = False
try:
    import timm

    assert timm.__version__ == "0.3.2"
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    DeiT_support = True
except:
    DeiT_support = False

if pl_bolts:
    ss_models_mapping = {
        "SimCLR": {
            "url": "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt",
            "transform": self_supervised.simclr.transforms.SimCLREvalDataTransform().online_transform,
        },
        # "SwAV":{"url":"https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar",
        #         "transform"}
    }
else:
    ss_models_mapping = {}


def get_last_layer_name(net):
    module = list(net._modules.items())[-1]
    if (
        type(module[1]) == torch.nn.modules.linear.Linear
        or type(module[1]) == torch.nn.modules.pooling.AdaptiveAvgPool2d
    ):
        return module[0]
    return f"{module[0]}:s:{get_last_layer_name(module[1])}"


def deep_get(dict_obj, key):
    d = dict_obj
    for k in key.split(":s:"):
        if (
            type(d) == torch.nn.modules.container.Sequential
            or type(d) == torchvision.models.resnet.Bottleneck
        ):
            d = d.__dict__["_modules"]
        d = d[k]
    return d


class Model_Operations:
    def __init__(self, model, layer_names):
        self.model = model
        self.outputs = []
        self.layer_names = layer_names
        for layer_name in layer_names:
            deep_get(self.model.__dict__["_modules"], layer_name).register_forward_hook(
                self.feature_hook
            )

    def feature_hook(self, module, input, output):
        self.outputs.append(output)

    def __call__(self, x):
        self.outputs = []
        _ = self.model(x)
        return dict(zip(self.layer_names, self.outputs))


class dataset_labeler(torchvision.datasets.DatasetFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (class_name, file_name, tensor_data)
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        sample_info = path.split("/")
        cls_name, file_name = sample_info[-2], sample_info[-1]
        return cls_name, file_name, sample


def main(args):
    logger = vastlogger.setup_logger(level=args.verbose)

    if args.self_supervised_approach is not None:
        logger.critical(
            f"Loding model from {ss_models_mapping[args.self_supervised_approach]['url']}"
        )
        model = (
            self_supervised.__dict__[args.self_supervised_approach]
            .load_from_checkpoint(
                ss_models_mapping[args.self_supervised_approach]["url"], strict=False
            )
            .encoder
        )
        val_transforms = ss_models_mapping[args.self_supervised_approach]["transform"]
    elif args.DeiT_model is not None:
        input_size = int(args.DeiT_model.split("_")[-1])
        size = int((256 / 224) * input_size)
        model = torch.hub.load(
            "facebookresearch/deit:main", args.DeiT_model, pretrained=True
        )
        val_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
    else:
        model = torchvision.models.__dict__[args.arch](pretrained=True)
        val_transforms = transforms.Compose(
            [
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    if args.weights is not None:
        state_dict = torch.load(args.weights, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if args.saved_with_data_parallel:
            new_state_dict = {}
            for k in state_dict:
                new_state_dict[k.replace("module.", "")] = state_dict[k]
            state_dict = new_state_dict

        if args.ignore_fc:
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]

        # specific to MoCoV2
        if any([k.startswith("module.encoder_q") for k in state_dict.keys()]):
            logger.critical(
                "\n\n\nAre you using a MoCo model? If not I may do something funky ahead"
            )
            if not args.dare_devil:
                temp = input("\nPlease confirm to continue or press Ctrl+C to exit\n")
            for k in list(state_dict.keys()):
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        logger.critical(f"\n\n\nMessage from model loading\n{msg}")

        if (
            len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0
        ) and not args.dare_devil:
            temp = input("\nPlease confirm to continue or press Ctrl+C to exit\n")

    logger.info(f"\n\n######### Model Architecture for {args.arch} ##############")
    logger.info(model)
    logger.info(f"######### Model Architecture for {args.arch} ##############\n\n")

    if args.layer_names is None:
        args.layer_names = [get_last_layer_name(model)]
    logger.critical(
        f"############# Will be extracting layer {args.layer_names} #############"
    )

    model.eval()
    model = model.to("cuda")
    modelObj = Model_Operations(model, args.layer_names)

    dataset_to_extract = dataset_labeler(
        args.dataset_path,
        pil_loader,
        extensions=(".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm"),
        transform=val_transforms,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_to_extract,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(mp.cpu_count() // 10, 1),
        pin_memory=False,
        drop_last=False,
    )
    pbar = tqdm(total=len(dataset_to_extract.classes))
    current_class = None
    current_class_im_name = []
    current_class_layer_outputs = []
    for _ in args.layer_names:
        current_class_layer_outputs.append([])

    hf = h5py.File(args.output_path, "w")

    with torch.no_grad():
        for i, (gt_class, img_name, images) in enumerate(data_loader):
            images = images.cuda()
            layer_outputs = modelObj(images)
            for layer in layer_outputs:
                layer_outputs[layer] = layer_outputs[layer].tolist()
            # Process all samples in the current batch
            for sample_no, (gt, im_name) in enumerate(zip(gt_class, img_name)):
                if current_class is None:
                    current_class = gt
                if len(current_class_im_name) > 0 and gt != current_class:
                    g = hf.create_group(current_class)
                    g.create_dataset(
                        "image_names",
                        data=np.array(
                            current_class_im_name,
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        ),
                    )
                    for layer_no, layer in enumerate(layer_outputs):
                        g.create_dataset(
                            layer, data=np.array(current_class_layer_outputs[layer_no])
                        )
                    pbar.update(1)

                    # Reset variables that hold data
                    current_class = gt
                    current_class_layer_outputs = []
                    for _ in args.layer_names:
                        current_class_layer_outputs.append([])
                    current_class_im_name = []

                current_class_im_name.append(im_name)
                for i, layer in enumerate(layer_outputs):
                    current_class_layer_outputs[i].append(layer_outputs[layer][sample_no])
        if len(current_class_im_name) > 0:
            g = hf.create_group(current_class)
            g.create_dataset(
                "image_names",
                data=np.array(
                    current_class_im_name, dtype=h5py.string_dtype(encoding="utf-8")
                ),
            )
            for layer_no, layer in enumerate(layer_outputs):
                g.create_dataset(
                    layer, data=np.array(current_class_layer_outputs[layer_no])
                )
            pbar.update(1)
    pbar.close()
    hf.close()


if __name__ == "__main__":
    pytorch_models = sorted(
        name
        for name in torchvision.models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torchvision.models.__dict__[name])
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script extracts features from a specific layer for a pytorch model",
    )
    parser.add_argument(
        "-v", "--verbose", help="To decrease verbosity increase", action="count", default=2
    )
    parser.add_argument(
        "--arch",
        default=None,
        choices=pytorch_models,
        help="The architecture from which to extract layers. "
        "Can be a model architecture already available in torchvision or a saved pytorch model.",
    )
    if DeiT_support:
        parser.add_argument(
            "--DeiT_model",
            default=None,
            choices=(
                "deit_tiny_patch16_224",
                "deit_small_patch16_224",
                "deit_base_patch16_224",
                "deit_tiny_distilled_patch16_224",
                "deit_small_distilled_patch16_224",
                "deit_base_distilled_patch16_224",
                "deit_base_patch16_384",
                "deit_base_distilled_patch16_384",
            ),
            help="DeiT model",
        )
    if pl_bolts:
        self_supervised_approaches = sorted(
            name
            for name in self_supervised.__dict__
            if not name.startswith("__") and callable(self_supervised.__dict__[name])
        )
        parser.add_argument(
            "--self_supervised_approach",
            default=None,
            choices=self_supervised_approaches,
            help="Self-supervised based model to load from pytorch lightning",
        )
    parser.add_argument(
        "--layer_names", nargs="+", help="Layer names to extract", default=None
    )
    parser.add_argument(
        "--dataset-path",
        help="directory containing the dataset in DatasetFolder format",
        default="/scratch/datasets/ImageNet/ILSVRC_2012/train",
        required=False,
    )
    parser.add_argument("--weights", help="network weights", default=None, required=False)
    parser.add_argument(
        "--output-path", help="output directory path", default="", required=True
    )
    parser.add_argument(
        "--batch-size", help="Number of samples per forward pass", default=256, type=int
    )
    parser.add_argument(
        "--saved-with-data-parallel",
        help="If you saved your model with data parallel set this flag",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ignore-fc",
        help="""
                                            Ignore FC layer useful only if number of classes in the loaded network
                                            is different from the standard network architecture and the layer being
                                            extracted is not the fc layer.
                                            """,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dare-devil",
        help="don't wait for user input even if you think they may be doing something wrong",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    if args.ignore_fc:
        assert (
            "fc" not in args.layer_names
        ), "OOPS! You have been stopped from doing something you might repent :P"

    if not pl_bolts:
        args.self_supervised_approach = None
    if not DeiT_support:
        args.DeiT_model = None
    main(args)
