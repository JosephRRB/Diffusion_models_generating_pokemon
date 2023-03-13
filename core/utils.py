import glob

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


# Dataset cloned from: https://github.com/HybridShivam/Pokemon
class ImageDataset(Dataset):
    def __init__(self, image_directory="image_data/", transform_fn=None):
        self.image_filenames = sorted(glob.glob(f"{image_directory}*"))
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform_fn:
            image = self.transform_fn(image)
        return image


def _create_image_loader(batch_size=2, image_size=64):
    transform_fn = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: 2 * img - 1)
    ])
    image_data = ImageDataset(transform_fn=transform_fn)
    loader = DataLoader(image_data, batch_size=batch_size, shuffle=True)
    return loader
