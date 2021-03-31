import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser("Convert folders of images to zipped numpy arrays")
parser.add_argument('--data_folder', default="", help="images folder")
parser.add_argument('--multiple', default=False, action="store_true", help="combine multiple compressed datasets")
parser.add_argument('--train', default=False, action="store_true", help="train or test set")
args = parser.parse_args()

root_folder = "/media/expansion1/navneedhmaudgalya/Datasets/cifar"

train_type = "train" if args.train else "test"

if args.multiple:
    bls_compressed_folders = ["", "_bls_001", "_bls_003", "_bls_005", "_bls_012", "_bls_033"]
    jpeg_compressed_folders = ["_jpeg_1", "_jpeg_5", "_jpeg_10", "_jpeg_20", "_jpeg_40"]

    all_images = []
    for i in tqdm(range(6)):
        data_folder = os.path.join(root_folder, train_type + bls_compressed_folders[i])

        data = [file for file in os.listdir(data_folder) if file.endswith(".png")]

        for filename in sorted(data, key = lambda x: int(x.split(".")[0])):
            image_path = os.path.join(data_folder,filename)
            img = np.array(Image.open(image_path).convert('RGB'))
            all_images.append(img)

    all_images = np.array(all_images)
    np.save(os.path.join(train_type + "_bls_all.npy"), all_images)
    print(all_images.shape)


else:
    data_folder = os.path.join(root_folder, train_type + args.data_folder)

    all_images = []
    files = [f for f in os.listdir(data_folder) if f.endswith(".png")]
    for filename in sorted(files, key = lambda x: int(x.split(".")[0])):
        image_path = os.path.join(data_folder,filename)
        img = np.array(Image.open(image_path).convert('RGB'))
        all_images.append(img)

    all_images = np.array(all_images)
    np.save(os.path.join("data/" + train_type + args.data_folder + ".npy"), all_images)
    print(all_images.shape)


