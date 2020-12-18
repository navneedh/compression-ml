from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class CifarDataset(Dataset):

    def __init__(self, data_folder, train = False, transform = None):
        self.root_dir = "/media/expansion1/navneedhmaudgalya/Datasets/cifar"

        if train == True:
            self.labels = np.load("{}/train.npy".format(self.root_dir))
            self.root_dir = os.path.join(self.root_dir, "train" + data_folder)
        else:
            self.labels = np.load("{}/test.npy".format(self.root_dir))
            self.root_dir = os.path.join(self.root_dir, "test" + data_folder)

        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, "{}.png".format(idx))
        img = Image.open(image_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]


# train_dset = CifarDataset(train = True, transform = transform)
# plt.imsave("test.png", train_dset[0][0].transpose(2,0))