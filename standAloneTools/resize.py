import cv2
import multiprocessing
from multiprocessing import Pool
import os
import argparse
import pathlib
from tqdm import tqdm

import numpy as np

parser = argparse.ArgumentParser(
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='This script is capable of resizing images in parallel, in order to avoid resizing during training.'
                                )
parser.add_argument("--org_dataset_path", help="Path to original images to resize",
                    default='/net/kato/store2/ImageNet/ILSVRC_2012/val_in_folders/')#,type=dir_path)
parser.add_argument("--resized_dataset_path", help="Path to resized images directory. Would be created if it doesn't exist",
                    default='/net/kato/store2/ImageNet/ILSVRC_2012/val/')#,type=dir_path)
parser.add_argument("--resized_image_height", help="resized_image_height", type=int, default=256)
parser.add_argument("--resized_image_width", help="resized_image_width", type=int, default=256)

args = parser.parse_args()

org_dataset_path = pathlib.Path(args.org_dataset_path)
resized_dataset_path = pathlib.Path(args.resized_dataset_path)
resized_dataset_path.mkdir(parents=True, exist_ok=True)

def process_each_file(file_name):
    img = cv2.imread(str(file_name))
    if img is None:
        print (f"Not a valid path {file_name}")
        return None
    image = cv2.resize(img,(args.resized_image_height,args.resized_image_width),interpolation=cv2.INTER_CUBIC)
    save_as = resized_dataset_path / str(file_name.parent).split('/')[-1] / file_name.name
    save_as.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_as),image)
    # If image original image was smaller than resized image
    if np.prod(img.shape)<np.prod(image.shape):
        return False
    else:
        return True


all_images_to_process = list(org_dataset_path.glob("*/*"))
print (f"Processing {len(all_images_to_process)} images")
p = Pool(int(multiprocessing.cpu_count()*10))
it = p.imap(process_each_file,all_images_to_process)
images_with_src_smaller_than_dest=0
with tqdm(total=len(all_images_to_process)) as progress_bar:
    for i in it:
        if i is not None and i:
            images_with_src_smaller_than_dest+=1
        progress_bar.update(1)
print (f"Total number of images for which src was smaller than destination {images_with_src_smaller_than_dest}/{len(all_images_to_process)}")