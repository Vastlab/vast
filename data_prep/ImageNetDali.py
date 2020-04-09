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
    def __init__(self, batch_size, device_id, num_gpus, csv_file):
        self.batch_size = batch_size
        with open(csv_file, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.files)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[
                     self.data_set_len * device_id // num_gpus: self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch_inputs = []
        batch_labels = []
        if self.i >= self.n:
            raise StopIteration
        for _ in range(self.batch_size):
            jpeg_path, label = self.files[self.i].split(',')
            f = open(jpeg_path, 'rb')
            batch_inputs.append(np.frombuffer(f.read(), dtype=np.uint8))
            batch_labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch_inputs, batch_labels)

    @property
    def size(self, ):
        return self.data_set_len
    next = __next__

class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=34)
        self.input = nvidia_ops.ExternalSource()
        self.input_label = nvidia_ops.ExternalSource()
        self.decode = nvidia_ops.ImageDecoder(device="mixed", output_type=nvidia_types.RGB)
        self.res = nvidia_ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cast = nvidia_ops.Cast(device="gpu", dtype=nvidia_types.FLOAT)
        self.cast2 = nvidia_ops.Cast(device="cpu", dtype=nvidia_types.FLOAT)
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = (self.cast(images) / 128) - 1
        self.labels_output = self.cast2(self.labels)
        return (output, self.labels_output)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration
