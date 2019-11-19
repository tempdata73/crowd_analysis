import cv2
import h5py
import torch
import random
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CrowdEstimationDataset(Dataset):

    def __init__(self, image_paths, augment=False, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        # Load density map
        target_path = image_path.replace('.jpg', '.h5').replace('images', 'densities')
        target_file = h5py.File(target_path)
        target = np.asarray(target_file['density'])

        # Fetch random patch on training
        if self.augment and random.randint(0, 1):
            image, target = self._get_random_crop(image, target)

        # Density map must be one eighth of original image
        width = int(target.shape[1] / 8)
        height = int(target.shape[0] / 8)
        target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC) * 64

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def _get_random_crop(self, image, target):
        center = (int(image.shape[1] / 2), int(image.shape[0] / 2))
        # Flip coin in order to get a corner or random patch
        if random.randint(0, 1):
            # Get corner
            dx = int(random.randint(0, 1) * center[0])
            dy = int(random.randint(0, 1) * center[1])
        else:
            # Get random patch
            dx = int(random.random() * center[0])
            dy = int(random.random() * center[1])

        cropped_image = image[dy: dy + center[1], dx: dx + center[0]]
        cropped_target = target[dy: dy + center[1], dx: dx + center[0]]

        # Flip coin to mirror image
        if random.randint(0, 1):
            # PyTorch currently does not support negative strides
            # Solution found in discuss.pytorch.org
            cropped_image = np.fliplr(cropped_image) - np.zeros_like(cropped_image)
            cropped_target = np.fliplr(cropped_target) - np.zeros_like(cropped_target)

        return cropped_image, cropped_target


def create_dataloader(image_paths, augment=False, **kwargs):
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformed_dataset = CrowdEstimationDataset(
        image_paths, augment=False, transform=preprocessing)
    dataloader = DataLoader(transformed_dataset, **kwargs)
    return dataloader


if __name__ == '__main__':
    import os
    import json
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Load image_paths
    ROOT_DIR = 'part_B'
    DATA_FILE = os.path.join(ROOT_DIR, 'train.json')
    with open(DATA_FILE) as infile:
        image_paths = json.load(infile)

    # Create dataset
    train_loader = create_dataloader(image_paths, batch_size=1, shuffle=True)
    iterator = iter(train_loader)
    image, target = next(iterator)

    # From tensor to numpy to image-ready
    image = np.transpose(image.squeeze().numpy(), (1, 2, 0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target = target.squeeze().numpy()
    print('Number of people = {:0.2f}'.format(target.sum()))

    # Show image
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(image)
    ax[0].set_title('Preprocessed image')
    ax[0].axis('off')

    ax[1].imshow(target, cmap=cm.jet)
    ax[1].set_title('Groundtruth density map')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
