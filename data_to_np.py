import numpy as np
import argparse
import os
from PIL import Image

parser = argparse.ArgumentParser("Convert folders of images to zipped numpy arrays")
parser.add_argument('--folder', default="train", help="images folder")
parser.add_argument('--summary', default=False, action="store_true")
args = parser.parse_args()

root_folder = "/media/expansion1/navneedhmaudgalya/Datasets/cifar"

data_folder = os.path.join(root_folder, args.folder)

all_images = []
for filename in sorted(os.listdir(data_folder), key = lambda x: int(x.split(".")[0])):
    image_path = os.path.join(data_folder,filename)
    img = np.array(Image.open(image_path).convert('RGB'))
    all_images.append(img)

all_images = np.array(all_images)
np.save(os.path.join("data/" + args.folder + ".npy"), all_images)
print(all_images.shape)


