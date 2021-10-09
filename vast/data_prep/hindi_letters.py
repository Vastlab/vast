import cv2
import glob
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np


def read_hindi_letters(file_name):
    img = cv2.imread(file_name, 0)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
    return img


class hindi_letters:
    def __init__(self):
        root_path = "/net/kato/datasets/Hindi_Letters/nhcd/"
        to_process = ["consonants/", "vowels/"]
        images = []
        p = ThreadPool(multiprocessing.cpu_count())
        for folder in to_process:
            files_to_process = glob.glob(root_path + folder + "*/*")
            raw_data = p.map(read_hindi_letters, files_to_process)
            images.extend(raw_data)
        self.images = np.array(images)[..., np.newaxis]
        self.images = self.images * (1.0 / 255.0)
        p.close()
        p.join()
        del p
