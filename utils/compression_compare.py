from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

"""
Compare different compression techniques with different similarity metrics
"""

def log10(x):
    return np.log(x)/np.log(10)

def psnr(im1, im2):
    mean_squared_error = mse(im1, im2)
    return 10 * log10(255**2/mean_squared_error)

def mse(im1, im2):
    return (np.square(im1 - im2)).mean()

def get_metrics(data_folder_1, data_folder_2, metric):
    metric_vals = []

    for i in tqdm(range(10000)):
        file_name = str(i) + ".png"
        file_path_1 = os.path.join(data_folder_1, file_name)
        file_path_2 = os.path.join(data_folder_2, file_name)


        image1 = np.array(Image.open(file_path_1).convert('RGB'))/255.

        image2 = (np.array(Image.open(file_path_2).convert('RGB'))/255.)[:,:,:3]
        if metric == "ssim":
            metric_vals.append(ssim(image1, image2, multichannel = True))
        elif metric == "psnr":
            metric_vals.append(psnr(image1, image2))
        elif metric == "mse":
            metric_vals.append(1 - mse(image1, image2))

    return metric_vals

def get_bpp(data_folder):
    return np.load(os.path.join(data_folder, "bpp.npy"))

def get_full_bpp(data_folder):
    return np.load(os.path.join(data_folder, "full_bpp.npy"))

if __name__ == "__main__":
    data_folder_1 = "data/cifar/test"
    data_folder_2 = "data/cifar/test_005"

    metric_vals = get_metrics(data_folder_1, data_folder_2, "ssim")

    plt.hist(metric_vals)
    plt.savefig("ssim.png")
