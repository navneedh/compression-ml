import argparse
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser("Convert folders of images to zipped numpy arrays")
parser.add_argument('--data_folder', default="test", help="images folder")
parser.add_argument('--out_folder', default="test", help="output folder")
parser.add_argument('--type', default='jpeg', help='jpeg, jpeg2, gn, sp, glp')
args = parser.parse_args()

data_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/" + args.data_folder
output_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/" + args.out_folder

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

num_data = 50000 if args.data_folder == "train" else 10000


if args.type == "jpeg":
	for i in tqdm(range(num_data)):
		file_name = "val_" + str(i) + ".png"
		output_file_path = os.path.join(output_folder, str(i) + ".png")
		input_file_path = os.path.join(data_folder, file_name)

		image = Image.open(input_file_path).convert("RGB")
		image.save(output_file_path, "JPEG", quality=60)


if args.type == "jpeg2":
	for i in tqdm(range(num_data)):
		file_name = str(i) + ".png"
		output_file_path = os.path.join(output_folder, str(i) + ".jp2")
		input_file_path = os.path.join(data_folder, file_name)

		image = Image.open(input_file_path).convert("RGB")
		image.save(output_file_path, 'JPEG2000', quality_mode='rates', quality_layers=[15])

#TODO: fix normalization (should just clip btwn 0 and 1) for adding gaussian noise to images
#TODO: debug implementation for salt and pepper
if args.type == "gn":
	for i in tqdm(range(num_data)):
		file_name = str(i) + ".png"
		output_file_path = os.path.join(output_folder, str(i) + ".png")
		input_file_path = os.path.join(data_folder, file_name)

		image = np.array(Image.open(input_file_path))/255.

		row,col,ch= image.shape
		mean = 0
		sigma = 0.01
		gauss = np.random.normal(mean,sigma,(row,col,ch))
		gauss = gauss.reshape(row,col,ch)
		image = image + gauss
		image = np.clip(image, 0, 1)


		plt.imsave(output_file_path, image)

if args.type == "sp":
	for i in tqdm(range(num_data)):
		file_name = str(i) + ".png"
		output_file_path = os.path.join(output_folder, str(i) + ".png")
		input_file_path = os.path.join(data_folder, file_name)

		image = np.array(Image.open(input_file_path))/255.

		row,col,ch = image.shape
		s_vs_p = 0.5
		amount = 0.04
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt))
		      for i in image.shape[:-1]]

		out[coords] = 1

		# Pepper mode
		num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper))
		      for i in image.shape[:-1]]
		out[coords] = 0

		plt.imsave(output_file_path, out)

if args.type == "glp":
	for i in tqdm(range(num_data)):
		file_name = str(i) + ".png"
		output_file_path = os.path.join(output_folder, str(i) + ".png")
		input_file_path = os.path.join(data_folder, file_name)

		image = cv2.imread(input_file_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		blur = cv2.GaussianBlur(image, 3)

		plt.imsave(output_file_path, blur)

if args.type == "cutout":
	pass

if args.type == "patchcutout":
	pass
