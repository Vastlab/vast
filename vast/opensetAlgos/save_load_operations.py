import time
import h5py
import math
import pathlib
import torch
import torch.multiprocessing as mp
from functools import partial
from typing import Generator, List, Dict, Tuple
from vast.tools import logger as vastlogger
from vast.DistributionModels import weibull

logger = vastlogger.get_logger()

class model_saver:
    def __init__(self, args,
                 process_combination: Tuple[str, int],
                 total_no_of_classes: int,
                 output_file_name: str = None
                 ) -> None:
        self.combination, self.process_rank = process_combination
        if output_file_name is None:
            model_file_name = pathlib.Path(f"{args.output_dir}/{self.combination}")
            model_file_name.mkdir(parents=True, exist_ok=True)
            model_file_name = model_file_name/pathlib.Path(f"{args.OOD_Algo}_model.hdf5")
        else:
            model_file_name = output_file_name
        logger.info(f"Saving model file at {model_file_name}")
        self.hf = h5py.File(model_file_name, "w")
        self.processed_classes = 0
        self.total_no_of_classes = total_no_of_classes

    def wait(self):
        while True:
            time.sleep(10)
            if self.processed_classes >= self.total_no_of_classes:
                break
        return

    def close(self):
        self.hf.close()
        logger.info(f"Closed model file successfully")

    def process_dict(self, group, model):
        for key_name in model:
            if type(model[key_name]) == dict:
                sub_group = group.create_group(f"{key_name}")
                self.process_dict(sub_group, model[key_name])
            else:
                group.create_dataset(f"{key_name}", data=model[key_name])
        return

    def __call__(self, cls_name, model):
        if model is not None:
            group = self.hf.create_group(f"{cls_name}")
            self.process_dict(group, model)
        else:
            logger.info(f" Class {cls_name} did not produce a model")
        self.processed_classes+=1
        if self.processed_classes%25==0:
            logger.info(f"Saved {self.combination} model for {self.processed_classes}/{self.total_no_of_classes} classes")


def model_loader(args, combination_str, training_data=None):
    model_file_name = pathlib.Path(f"{args.output_dir}/{combination_str}/{args.OOD_Algo}_model.hdf5")
    logger.info(f"Loading model file {model_file_name}")
    model_dict={}
    with h5py.File(model_file_name, "r") as hf:
        for cls in hf.keys():
            model_dict[cls]={}
            for k in hf[cls].keys():
                if k not in ['weibull', 'weibulls']:
                    model_dict[cls][k] = torch.tensor(hf[cls][k][()])
                else:
                    weibull_dict = {}
                    for param in hf[cls][k].keys():
                        weibull_dict[param] = torch.tensor(hf[cls][k][param][()])
                    model_dict[cls][k] =  weibull.weibull(weibull_dict)
            if args.OOD_Algo in ['EVM','Turbo_EVM'] and 'extreme_vectors' not in [*model_dict[cls]]:
                if training_data is None:
                    logger.error(f"Looks like you do not have the 'extreme_vectors' saved, "
                                 f"please provide the training features so extreme_vectors_indexes can be used")
                else:
                    model_dict[cls]['extreme_vectors'] = training_data[cls][model_dict[cls]['extreme_vectors_indexes'], :]
    logger.info(f"Loaded OOD model from {model_file_name}")
    return model_dict