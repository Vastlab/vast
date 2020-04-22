import csv
from .SAILON import *
import numpy as np
import random
random.seed(0)
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as nvidia_ops
import nvidia.dali.types as nvidia_types

# Don't need to inherit from 'object' in Python3 onwards
# Only needed for backward compatibility: https://stackoverflow.com/questions/4015417/python-class-inherits-object
class ExternalInputIterator():
    def __init__(self, batch_size, device_id, num_gpus, filenames, images_path,
                 debug=False, random_seed=None, unknowns_label=-1):
        self.batch_size = batch_size
        self.files = []
        for file_no,filename in enumerate(filenames):
            if filename.suffix == '.csv':
                with open(filename, 'r') as f:
                    files = list(csv.reader(f))
            elif filename.suffix == '.json':
                files = get_files_labels(filename)
            if file_no == 1:
                names,labels = zip(*files)
                files = list(zip(names,[unknowns_label]*len(names)))
            self.files.extend(files)

        if debug:
            self.files = self.files[:(2*batch_size*num_gpus)]
        # whole data set size
        self.data_set_len = len(self.files)
        # shuffle samples for training set
        if random_seed is not None:
            random.seed(random_seed)
            shuffle(self.files)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[
                     self.data_set_len * device_id // num_gpus: self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)
        self.images_path = images_path

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch_inputs = []
        batch_labels = []
        while len(batch_inputs)<self.batch_size and self.i<self.n:
            jpeg_path, label = self.files[self.i]
            f = open(self.images_path+jpeg_path, 'rb')
            batch_inputs.append(np.frombuffer(f.read(), dtype=np.uint8))
            batch_labels.append(np.array([label], dtype=np.int32))
            self.i += 1
        if self.i >= self.n:
            raise StopIteration
        return (batch_inputs, batch_labels)

    @property
    def size(self, ):
        return self.data_set_len
    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, device_type="gpu",
                 training=False, image_size=224):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=34,prefetch_queue_depth={ "cpu_size": 10, "gpu_size": 2})
        self.input = nvidia_ops.ExternalSource()
        self.input_label = nvidia_ops.ExternalSource()
        self.training = training
        self.crop_loc = nvidia_ops.Uniform(range=(0.,1.))
        self.coin = nvidia_ops.CoinFlip(probability=0.5)
        self.rotation_angle = nvidia_ops.Uniform(range=(0.,10.))
        self.decode = nvidia_ops.ImageDecoder(device="mixed" if device_type=="gpu" else "cpu", output_type=nvidia_types.RGB)
        self.rotate = nvidia_ops.Rotate(device=device_type)
        self.resize = nvidia_ops.Resize(device=device_type, resize_shorter=(image_size*256/224))
        self.crop_mirror_normalize = nvidia_ops.CropMirrorNormalize(device=device_type, crop=(image_size,image_size),
                                                                    mean=128,std=128,output_layout='HWC')
        self.jitter = nvidia_ops.Jitter(device="gpu", nDegree=4)
        self.transpose = nvidia_ops.Transpose(device="gpu",perm=(2,0,1))
        self.cast = nvidia_ops.Cast(device="gpu", dtype=nvidia_types.FLOAT)
        self.external_data = external_data
        self.iterator = iter(self.external_data)
        self.device_type = device_type

    def training_data_augmentation(self, images):
        images = self.rotate(images, angle=self.rotation_angle())
        images = self.resize(images)
        images = self.crop_mirror_normalize(images, crop_pos_x=self.crop_loc(), crop_pos_y=self.crop_loc(),
                                            mirror=self.coin())
        if self.device_type!="gpu":
            images = images.gpu()
        # Does not work https://github.com/NVIDIA/DALI/issues/966
        # images = self.jitter(images)
        return images

    def validation_data_augmentation(self, images):
        images = self.resize(images)
        images = self.crop_mirror_normalize(images)
        if self.device_type!="gpu":
            images = images.gpu()
        return images

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        if self.training:
            images = self.training_data_augmentation(images)
        else:
            images = self.validation_data_augmentation(images)
        images = self.transpose(images)
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration
