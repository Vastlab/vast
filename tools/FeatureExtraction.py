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
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
def get_last_layer_name(net):
    module=list(net._modules.items())[-1]
    if type(module[1])==torch.nn.modules.linear.Linear or \
            type(module[1])==torch.nn.modules.pooling.AdaptiveAvgPool2d:
        return module[0]
    return f"{module[0]}:s:{get_last_layer_name(module[1])}"

def deep_get(dict_obj, key):
    d = dict_obj
    for k in key.split(":s:"):
        if type(d) == torch.nn.modules.container.Sequential or type(d) == torchvision.models.resnet.Bottleneck:
            d = d.__dict__['_modules']
        d = d[k]
    return d

class Model_Operations():
    def __init__(self, model, layer_names):
        self.model = model
        self.outputs = []
        self.layer_names = layer_names
        for layer_name in layer_names:
            deep_get(self.model.__dict__['_modules'], layer_name).register_forward_hook(self.feature_hook)

    def feature_hook(self, module, input, output):
        self.outputs.append(output)

    def __call__(self, x):
        self.outputs = []
        _ = self.model(x)
        return dict(zip(self.layer_names,self.outputs))


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
        sample_info=path.split("/")
        cls_name,file_name=sample_info[-2],sample_info[-1]
        return cls_name, file_name, sample


def main(args):
    # cudnn.benchmark = True
    # Data loading code
    pytorch_models = sorted(name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))
    if args.arch in pytorch_models:
        model = models.__dict__[args.arch](pretrained=True)
    # Currently only specific to MoCoV2
    if args.weights is not None:
        state_dict = torch.load(args.weights, map_location="cpu")['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"msg {msg}")

    print(f"\n\n######### Model Architecture for {args.arch} ##############")
    print(model)
    print(f"######### Model Architecture for {args.arch} ##############\n\n")

    if args.layer_names is None:
        args.layer_names=[get_last_layer_name(model)]
        print(f"############# Will be extracting layer {args.layer_names} #############")

    model.eval()
    model = model.to('cuda')
    modelObj = Model_Operations(model, args.layer_names)

    dataset_to_extract=dataset_labeler(args.dataset_path, pil_loader,
                                       extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm'),
                                       transform=val_transforms)
    data_loader = torch.utils.data.DataLoader(dataset_to_extract,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=mp.cpu_count(),
                                              pin_memory=False,
                                              drop_last=False)
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
            for sample_no,(gt,im_name) in enumerate(zip(gt_class, img_name)):
                if current_class is None: current_class=gt
                if len(current_class_im_name)>0 and gt != current_class:
                    g = hf.create_group(current_class)
                    g.create_dataset('image_names', data=np.array(current_class_im_name, dtype=h5py.string_dtype(encoding='utf-8')))
                    for layer_no, layer in enumerate(layer_outputs):
                        g.create_dataset(layer, data=np.array(current_class_layer_outputs[layer_no]))
                    pbar.update(1)

                    # Reset variables that hold data
                    current_class = gt
                    current_class_layer_outputs = []
                    for _ in args.layer_names:
                        current_class_layer_outputs.append([])
                    current_class_im_name = []

                current_class_im_name.append(im_name)
                for i,layer in enumerate(layer_outputs):
                    current_class_layer_outputs[i].append(layer_outputs[layer][sample_no])
        if len(current_class_im_name) > 0:
            g = hf.create_group(current_class)
            g.create_dataset('image_names',
                             data=np.array(current_class_im_name, dtype=h5py.string_dtype(encoding='utf-8')))
            for layer_no, layer in enumerate(layer_outputs):
                g.create_dataset(layer, data=np.array(current_class_layer_outputs[layer_no]))
            pbar.update(1)
    pbar.close()
    hf.close()

if __name__ == '__main__':
    pytorch_models = sorted(name for name in torchvision.models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torchvision.models.__dict__[name]))

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="This script extracts features from a specific layer for a pytorch model")
    parser.add_argument("--arch",
                        default='resnet18', choices=pytorch_models,
                        help="The architecture from which to extract layers. "
                             "Can be a model architecture already available in torchvision or a saved pytorch model.")
    parser.add_argument("--layer_names",
                        nargs="+",
                        help="Layer names to extract",
                        default=None)
    parser.add_argument("--dataset-path", help="directory containing the dataset in DatasetFolder format",
                        default="/scratch/datasets/ImageNet/ILSVRC_2012/train", required=False)
    parser.add_argument("--weights", help="network weights", default=None, required=False)
    parser.add_argument("--output-path", help="output directory path", default="", required=True)
    parser.add_argument("--batch-size", help="Number of samples per forward pass", default=256, type=int)
    args = parser.parse_args()

    main(args)
