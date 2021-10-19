import glob
import numpy as np
import cv2
import multiprocessing
from multiprocessing.pool import ThreadPool


def read_NOT_MNIST(file_name):
    img = cv2.imread(file_name, 0)
    return img


class NOT_MNIST:
    def __init__(self, invert_image=True):
        root_path = "/net/kato/datasets/notMNIST_small/"
        files_to_process = glob.glob(root_path + "*/*.png")
        p = ThreadPool(multiprocessing.cpu_count())
        images = p.map(read_NOT_MNIST, files_to_process)
        self.images = np.array(images)[..., np.newaxis]
        self.images = self.images * (1.0 / 255.0)
        if invert_image:
            self.images = 1 - self.images
        p.close()
        p.join()
        del p
