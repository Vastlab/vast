import csv
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
        self.anchor_loc = nvidia_ops.Uniform(range=(0.,1.),shape =(4,))
        self.shape_loc = nvidia_ops.Uniform(range=(0.2,0.5),shape =(4,))
        self.coin = nvidia_ops.CoinFlip(probability=0.5)
        self.rotation_angle = nvidia_ops.Uniform(range=(-45.,45.))
        self.train_decode = nvidia_ops.ImageDecoderRandomCrop(device="mixed" if device_type=="gpu" else "cpu",
                                                              device_memory_padding = 211025920,
                                                              host_memory_padding = 140544512,
                                                              output_type=nvidia_types.RGB,
                                                              random_aspect_ratio=[0.5,4.],
                                                              random_area=[0.01,1.0],
                                                              num_attempts=100)
        self.val_decode = nvidia_ops.ImageDecoder(device="mixed" if device_type=="gpu" else "cpu", output_type=nvidia_types.RGB)
        self.rotate = nvidia_ops.Rotate(device=device_type)
        self.erase = nvidia_ops.Erase(device=device_type, axis_names="HW", centered_anchor=True, normalized=True, fill_value=0.5)
        self.resize = nvidia_ops.Resize(device=device_type, resize_shorter=(image_size*256/224))
        self.crop_mirror_normalize = nvidia_ops.CropMirrorNormalize(device=device_type,
                                                                    crop=(image_size,image_size),
                                                                    mean=[0.485*255, 0.456*255, 0.406*255],
                                                                    std=[0.229*255, 0.224*255, 0.225*255],
                                                                    output_layout='HWC')
        self.jitter = nvidia_ops.Jitter(device="gpu", nDegree=4)
        self.transpose = nvidia_ops.Transpose(device="gpu",perm=(2,0,1))
        self.cast = nvidia_ops.Cast(device="gpu", dtype=nvidia_types.FLOAT)
        self.external_data = external_data
        self.iterator = iter(self.external_data)
        self.device_type = device_type

    def training_data_augmentation(self):
        images = self.train_decode(self.jpegs)
        images = self.rotate(images, angle=self.rotation_angle())
        images = self.resize(images)
        images = self.crop_mirror_normalize(images, crop_pos_x=self.crop_loc(), crop_pos_y=self.crop_loc(),
                                            mirror=self.coin())
        images = self.erase(images, anchor=self.anchor_loc(), shape=self.shape_loc())
        if self.device_type!="gpu":
            images = images.gpu()
        images = self.jitter(images)
        return images

    def validation_data_augmentation(self):
        images = self.val_decode(self.jpegs)
        images = self.resize(images)
        images = self.crop_mirror_normalize(images)
        if self.device_type!="gpu":
            images = images.gpu()
        return images

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        if self.training:
            images = self.training_data_augmentation()
        else:
            images = self.validation_data_augmentation()
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
