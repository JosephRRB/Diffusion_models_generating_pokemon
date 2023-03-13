import glob

import matplotlib.pyplot as plt

from torchvision.io import read_image, ImageReadMode
from PIL import Image
from torch.utils.data import Dataset


# Dataset cloned from: https://github.com/HybridShivam/Pokemon
class ImageDataset(Dataset):
    def __init__(self, image_directory="images/", transform_fn=None):
        self.image_filenames = sorted(glob.glob(f"{image_directory}*"))
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        # image = read_image(image_path, mode=ImageReadMode.RGB)
        image = Image.open(image_path).convert('RGB')
        if self.transform_fn:
            image = self.transform_fn(image)
        return image
