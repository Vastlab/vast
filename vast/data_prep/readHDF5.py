"""
Reads HDF5 files (provided in feature_files argument) in parallel, using the key (provided in the layer_names argument).
If a list of files is provided to feature_files argument, then their features are concatenated
i.e. two files with n and m dimensional samples respectively produces n+m dimensional samples
"""

from vast.tools import logger as vastlogger
import h5py
import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import torch.multiprocessing as mp
from functools import partial
import numpy as np
from pathlib import Path

logger = vastlogger.get_logger()


def params(parser):
    data_prep_params = parser.add_argument_group("Pre-extracted feature details")
    data_prep_params.add_argument(
        "--training_feature_files",
        nargs="+",
        help="HDF5 file path for training data",
        default=[
            "/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_train.hdf5"
        ],
    )
    data_prep_params.add_argument(
        "--validation_feature_files",
        nargs="+",
        help="HDF5 file path for validation data",
        default=[
            "/net/reddwarf/bigscratch/adhamija/Features/MOCOv2/imagenet_1000_val.hdf5"
        ],
    )
    data_prep_params.add_argument(
        "--layer_names", nargs="+", help="Layer names to train EVM on", default=["avgpool"]
    )
    return parser, data_prep_params


def read_features(args, feature_file_names=None, cls_to_process=None):
    try:
        h5_objs = [h5py.File(file_name, "r") for file_name in feature_file_names]
        file_layer_comb = list(zip(h5_objs, args.layer_names))
        if cls_to_process is None:
            cls_to_process = sorted(list(h5_objs[0].keys()))
        if args.debug:
            cls_to_process = cls_to_process[:50]
        for cls in cls_to_process:
            temp = []
            for hf, layer_name in file_layer_comb:
                temp.append(torch.squeeze(torch.tensor(hf[cls][layer_name])))
            if "image_names" in [*hf[cls]]:
                image_names = hf[cls]["image_names"][()].tolist()
            else:
                image_names = None
            if len(temp[0].shape) == 1:
                temp[0] = temp[0][None,:]
            features = torch.cat(temp,dim=0)
            yield cls, features, image_names
    finally:
        for h in h5_objs:
            h.close()


def prep_single_chunk(args, cls_to_process):
    features_gen = read_features(
        args, feature_file_names=args.feature_files, cls_to_process=cls_to_process
    )
    data_to_return = {}
    for cls, feature, image_names in features_gen:
        data_to_return[cls] = {}
        if image_names is not None:
            data_to_return[cls]["images"] = np.array(
                [f"{cls}/{_.decode('ascii')}" for _ in image_names], dtype="<U30"
            )
        data_to_return[cls]["features"] = feature.share_memory_()
    return data_to_return


@vastlogger.time_recorder
def prep_all_features_parallel(args, all_class_names=None, use_multiprocessing=True):
    for f in args.feature_files:
        assert Path(f).is_file(), f"File {f} does not exist"
    if all_class_names is None:
        with h5py.File(args.feature_files[0], "r") as hf:
            all_class_names = sorted([*hf])
            logger.info("Following keys are available for each class in the HDF5 file")
            for k in [*hf[all_class_names[-1]]]:
                logger.info(f"{k} {hf[all_class_names[-1]][k].shape}")
            logger.info(f"Will be extracting {args.layer_names}")
            assert (
                len(set(args.layer_names) - set([*hf[all_class_names[-1]]])) == 0
            ), logger.error("mismatch in layer to find vs layers present")
            for l in args.layer_names:
                logger.info(
                    f"Original shape of {l} is {hf[all_class_names[-1]][l].shape} "
                    f"Resizing it to {[*torch.squeeze(torch.tensor(hf[all_class_names[-1]][l])).shape]}"
                )
    cls_per_chunk = max(len(all_class_names) // (mp.cpu_count() - 30), 5)
    if args.debug:
        all_class_names = all_class_names[:100]
        cls_per_chunk = 2
    if use_multiprocessing:
        all_class_batches = [
            all_class_names[i : i + cls_per_chunk]
            for i in range(0, len(all_class_names), cls_per_chunk)
        ]
        p = mp.Pool(min(30, mp.cpu_count()))
        all_data = p.imap_unordered(partial(prep_single_chunk, args), all_class_batches)
        all_classes = {}
        for data_returned in all_data:
            all_classes.update(data_returned)
    else:
        all_classes = prep_single_chunk(args, all_class_names)
    return all_classes
