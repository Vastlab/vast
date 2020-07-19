import random
random.seed(0)
from random import shuffle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class ImageNetPytorch(Dataset):
    def __init__(self, csv_file, images_path, debug=False, shuffle_samples=False,
                transform=[
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[128,128,128],std=[128,128,128])]
                ):
        with open(csv_file, 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        if debug:
            self.files = self.files[:1000]
        if shuffle_samples:
            shuffle(self.files)
        self.images_path = images_path
        self.transform = transforms.Compose(transform)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        jpeg_path, label = self.files[self.i].split(',')
        img = Image.open(self.images_path+jpeg_path).convert('RGB')
        x = self.transform(img)
        y = torch.as_tensor(self.label, dtype=torch.int64)
        return (x,y)
