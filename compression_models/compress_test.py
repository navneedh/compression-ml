import tensorflow as tf
import os
import zipfile
import collections
from PIL import Image
import tfci
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Compress dataset using pretrained compression models')
parser.add_argument('--dataset', type=str, default="cifar/test",
                    help='folder containing data to compress')
parser.add_argument('--output', default="test",
                    help='folder to save compressed images')
parser.add_argument('--model_name', default="hific-mi",
                    help='pretrained compression model')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



args = parser.parse_args()

def get_bpp(image_dimensions, num_bytes):
  w, h = image_dimensions
  return num_bytes * 8 / (w * h)

def has_alpha(img_p):
  im = Image.open(img_p)
  return im.mode == 'RGBA'

OUT_DIR = os.path.join("compression_models/output", args.output)
FILES_DIR = os.path.join("data", args.dataset)
MODEL = args.model_name
TMP_OUT = 'out.tfci'

if not os.path.isdir(OUT_DIR):
  os.mkdir(OUT_DIR)

# all_files = [str(i) + ".png" for i in range(5)]
# data_bytes = []
# for file_name in tqdm(all_files):
#   if os.path.isdir(file_name):
#     continue
#   full_path = os.path.join(FILES_DIR, file_name)
#   # if has_alpha(full_path):
#   #   print('Skipping because of alpha channel:', file_name)
#   #   continue
#   file_name, _ = os.path.splitext(file_name)

#   output_path = os.path.join(OUT_DIR,file_name + ".png")
#   # if os.path.isfile(output_path):
#   #   print('Skipping', output_path, '-- exists already.')
#   #   continue

#   # Old Approach
#   tfci.compress(MODEL, full_path, TMP_OUT)
#   num_bytes = os.path.getsize(TMP_OUT)
#   print(num_bytes)
#   tfci.decompress(TMP_OUT, output_path)
#   data_bytes.append(num_bytes)

# np.save(os.path.join(OUT_DIR,"data_bytes.npy"), data_bytes)
all_output_path = [os.path.join(OUT_DIR, str(i) + ".png") for i in range(10000)]
all_files_path = [os.path.join(FILES_DIR, str(i) + ".png") for i in range(10000)]
tfci.dataset_compressor(MODEL, all_files_path, all_output_path)
