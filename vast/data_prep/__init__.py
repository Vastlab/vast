import gzip
import struct
import numpy as np

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def read_labels(fname):
    with gzip.open(fname, 'rb') as f:
        # reads 2 big-ending integers
        magic_nr, n_examples = struct.unpack(">II", f.read(8))
        # reads the rest, using an uint8 dataformat (endian-less)
        labels = np.fromstring(f.read(), dtype='uint8')
        return labels

def read_images(fname):
    with gzip.open(fname, 'rb') as f:
        # reads 4 big-ending integers
        magic_nr, n_examples, rows, cols = struct.unpack(">IIII", f.read(16))
        shape = (n_examples, rows*cols)
        # reads the rest, using an uint8 dataformat (endian-less)
        images = np.fromstring(f.read(), dtype='uint8').reshape(shape)
        images = 255.-images
        return images

#from .mnist import mnist
#from .letters import letters
#from .hindi_letters import hindi_letters
#from .lfw import LFW
#from .GridMNIST import GridMNIST
from .ImageNetDali import *