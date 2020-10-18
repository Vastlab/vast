import argparse
import os
import numpy as np
import h5py
import pathlib
import multiprocessing as mp
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

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

class ImageNetPytorch(Dataset):
    def __init__(self, csv_file, images_path,
                 debug=False, shuffle_samples=False,
                 transform=[transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]
                ):
        with open(csv_file, 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        if debug:
            self.files = self.files[:1000]
        self.files = sorted(self.files)
        if '360' in csv_file:
            images_path = images_path + '/ILSVRC_2010-360classes/'
        else:
            images_path = images_path + '/ILSVRC_2012/'
        self.images_path = images_path
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        jpeg_path, _ = self.files[index].split(',')
        img = Image.open(self.images_path+jpeg_path).convert('RGB')
        x = self.transform(img)
        return (x,jpeg_path.split("/")[-1], jpeg_path.split("/")[-2])


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

    model.eval()
    model = model.to('cuda')
    modelObj = Model_Operations(model, args.layer_names)


    output_file_path = pathlib.Path(f"{args.output_path}/{args.arch}/")
    output_file_path.mkdir(parents=True, exist_ok=False)
    for file_name in ['imagenet_1000_train.csv','imagenet_1000_val.csv','imagenet_360.csv']:
        if '360' in file_name:
            pbar = tqdm(total=360)
        else:
            pbar = tqdm(total=1000)
        data_loader = torch.utils.data.DataLoader(ImageNetPytorch(args.input_csv_path + file_name,
                                                                  args.images_path),
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=1*mp.cpu_count(),
                                                  pin_memory=False)
        current_class = None
        current_class_im_name = []
        current_class_layer_outputs = []
        for _ in args.layer_names:
            current_class_layer_outputs.append([])

        hf = h5py.File(f"{output_file_path}/{file_name.split('.csv')[0]}.hdf5", "w")

        with torch.no_grad():
            for i, (images, img_name, gt_class) in enumerate(data_loader):
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
                for layer in layer_outputs:
                    g.create_dataset(layer, data=layer_outputs[layer])
                pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="This script extracts features from a specific layer for a pytorch model")
    parser.add_argument("--arch",
                        default='resnet18',
                        help="The architecture from which to extract layers. "
                             "Can be a model architecture already available in torchvision or a saved pytorch model.")
    parser.add_argument("--layer_names",
                        nargs="+",
                        help="Layer names to extract",
                        default=["fc"])
    parser.add_argument("--images-path", help="directory containing imagenet images",
                        default="/net/ironman/scratch/datasets/ImageNet/", required=False)
    parser.add_argument("--weights", help="network weights",
                        default=None, required=False)
    parser.add_argument("--input-csv-path", help="directory path containing imagenet csvs",
                        default="/home/jschwan2/simclr-converter/", required=False)
    parser.add_argument("--output-path", help="output directory path", default="", required=True)
    parser.add_argument("--batch-size", help="Number of samples per forward pass", default=256, type=int)
    args = parser.parse_args()

    """
    pytorch_models = sorted(name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))
    for pm in pytorch_models:
        if 'mnasnet' in pm:
            args.arch = pm
    """
    main(args)
