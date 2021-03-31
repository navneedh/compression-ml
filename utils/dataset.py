from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# class CifarDataset(Dataset):
#
#     def __init__(self, data_folder, train = False, transform = None):
#         self.root_dir = "data"
#         self.comp_levels = np.array([])
#         self.train = train
#
#         if data_folder == "_bls_all":
#             if train == True:
#                 self.labels = np.tile(np.load("{}/train_labels.npy".format(self.root_dir)), 5)
#                 self.images = np.load(os.path.join(self.root_dir, "train" + data_folder + ".npy"))
#                 self.comp_levels = np.array([[i] * 50000 for i in range(1,6)]).flatten()
#             else:
#                 self.labels = np.tile(np.load("{}/test_labels.npy".format(self.root_dir)), 5)
#                 self.images = np.load(os.path.join(self.root_dir, "test" + data_folder + ".npy"))
#                 self.comp_levels = np.array([[i] * 10000 for i in range(1,6)]).flatten()
#
#         else:
#             if train == True:
#                 self.labels = np.load("{}/train_labels.npy".format(self.root_dir))
#                 self.images = np.load(os.path.join(self.root_dir, "train" + data_folder + ".npy"))
#             else:
#                 self.labels = np.load("{}/test_labels.npy".format(self.root_dir))
#                 self.images = np.load(os.path.join(self.root_dir, "test" + data_folder + ".npy"))
#
#         self.transform = transform
#
#     def __len__(self):
#         return self.labels.shape[0]
#
#     def __getitem__(self, idx):
#
#         img = self.images[idx]
#
#         if self.transform:
#             img = self.transform(img)
#
#         if len(self.comp_levels) > 0:
#             return img, self.labels[idx]
#
#         return img, self.labels[idx]
#
#     def filter(self, filter_comps):
#         assert min(filter_comps) > 0 and max(filter_comps) < 6
#
#         filtered_images = []
#         if len(self.comp_levels) == 0:
#             raise ValueError("Please specify data folder with all compressed data")
#         else:
#             self.comp_levels = np.array([[i] * 10000 for i in filter_comps]).flatten()
#
#             for comp in filter_comps:
#                 if self.train:
#                     filtered_images.append(self.images[(comp - 1) * 50000: comp * 50000])
#                     self.labels = self.labels[0:len(filter_comps) * 50000]
#                 else:
#                     filtered_images.append(self.images[(comp - 1) * 10000: comp * 10000])
#                     self.labels = self.labels[0:len(filter_comps) * 10000]
#
#             self.images = np.concatenate(filtered_images)



class ClassificationDataset(Dataset):

    def __init__(self, data_folder, dataset, train = False, transform = None,):
        if dataset == "cifar":
            self.root_dir = "/media/expansion1/navneedhmaudgalya/Datasets/cifar"
            if train == True:
                self.labels = np.load("{}/train_labels.npy".format(self.root_dir))
                self.root_dir = os.path.join(self.root_dir, "train" + data_folder)
            else:
                self.labels = np.load("{}/test_labels.npy".format(self.root_dir))
                self.root_dir = os.path.join(self.root_dir, "test" + data_folder)

        elif dataset == "tiny":
            self.root_dir = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet"
            self.images = []

            if train == True:
                self.labels = pickle.load(open(os.path.join(self.root_dir, "train_imageidx_to_labelidx.p"), "rb"))
                self.root_dir = os.path.join(self.root_dir, "train" + data_folder)
                for idx in range(100000):
                    image_path = os.path.join(self.root_dir, "{}.png".format(idx))
                    img = Image.open(image_path).convert('RGB')
                    self.images.append(img)
            else:
                self.labels = pickle.load(open(os.path.join(self.root_dir, "test_imageidx_to_labelidx.p"), "rb"))
                self.root_dir = os.path.join(self.root_dir, "test" + data_folder)
                for idx in range(10000):
                    image_path = os.path.join(self.root_dir, "{}.png".format(idx))
                    img = Image.open(image_path).convert('RGB')
                    self.images.append(img)


        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(list(self.labels))

    def __getitem__(self, idx):
        if self.dataset == "cifar":
            image_path = os.path.join(self.root_dir, "{}.png".format(idx))
            img = Image.open(image_path).convert('RGB')

        elif self.dataset == "tiny":
            img = self.images[idx]

        if self.transform:
            img = self.transform(img)


        return img, self.labels[idx]



#

# train_dset = CifarDataset(train = True, transform = transform)
# plt.imsave("test.png", train_dset[0][0].transpose(2,0))