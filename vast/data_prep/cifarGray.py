import numpy as np
import cv2
import pickle


class cifar_prep:
    def __init__(self):
        cifar_data = pickle.load(
            open("/net/kato/datasets/cifar-10-batches-py/data_batch_1", "rb")
        )
        raw_images = cifar_data["data"].reshape(
            cifar_data["data"].shape[0], 32, 32, 3, order="F"
        )
        resized_images = []
        for img in raw_images:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_images.append(
                cv2.resize(gray_img, (28, 28), interpolation=cv2.INTER_CUBIC)
            )
        resized_images = np.array(resized_images)[..., np.newaxis]
        self.images = resized_images * (1.0 / 255.0)
