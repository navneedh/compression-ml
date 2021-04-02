# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.

This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704

With patches from Victor Xing <victor.t.xing@gmail.com>

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

import argparse
import glob
import sys

from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from imageio import imwrite

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = tf.squeeze(image, 0)
  if image.dtype.is_floating:
    image = tf.round(image)
  if image.dtype != tf.uint8:
    image = tf.saturate_cast(image, tf.uint8)
  string = tf.image.encode_png(image)
  return tf.io.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            3, (9, 9), name="layer_2", corr=False, strides_up=4,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor



def compress_cifar(args):
  """Compresses an image."""

  output_folder = "/media/expansion1/navneedhmaudgalya/Datasets/cifar/train_bls_1n"
  # output_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/test_tiny_bls_001"

  if not os.path.exists(output_folder):
      os.mkdir(output_folder)

  bpp = []
  full_bpp = []
  compressed_imgs = []

  # Load input image and add batch dimension.
  index = tf.placeholder(tf.string)
  # image_file_name = "{}.png".format(index.eval())
  # image_file_path = os.path.join("../data/cifar/test/", image_file_name)

  x = read_png(index)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Transform and compress the image.
  y = analysis_transform(x)
  string = entropy_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat_orig = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat_orig = tf.clip_by_value(x_hat_orig, 0, 1)
  x_hat = tf.round(x_hat_orig * 255)

  # mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  # psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  # msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
    for i in tqdm(range(50000)):
        image_file_name = "{}.png".format(i)
        image_file_path = str(os.path.join("/media/expansion1/navneedhmaudgalya/Datasets/cifar/train/", image_file_name))
        # op = write_png("test_005/{}.png".format(i), x_hat)
        x_h, arrays, inf_bpp = sess.run([x_hat, tensors, eval_bpp], feed_dict={index: image_file_path})
        plt.imsave("{}/{}".format(output_folder, image_file_name), x_h[0]/255.)
        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)

        bpp.append(inf_bpp)
        full_bpp.append(len(packed.string) * 8 / (32 * 32))
        compressed_imgs.append(packed.string)

        # sess.run(op, feed_dict={index: image_file_path})

  np.save("{}/bpp.npy".format(output_folder), bpp)
  np.save("{}/full_bpp.npy".format(output_folder), full_bpp)
  pickle.dump(compressed_imgs, open("{}/compressed_imgs.p".format(output_folder), "wb"))


def compress_tiny(args):
  """Compresses an image."""

  output_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/test_bls_05"

  if not os.path.exists(output_folder):
      os.mkdir(output_folder)

  bpp = []
  full_bpp = []
  compressed_imgs = []

  # Load input image and add batch dimension.
  index = tf.placeholder(tf.string)
  # image_file_name = "{}.png".format(index.eval())
  # image_file_path = os.path.join("../data/cifar/test/", image_file_name)

  x = read_png(index)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Transform and compress the image.
  y = analysis_transform(x)
  string = entropy_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat_orig = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat_orig = tf.clip_by_value(x_hat_orig, 0, 1)
  x_hat = tf.round(x_hat_orig * 255)

  # mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  # psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  # msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]

    data_folder = "/media/expansion1/navneedhmaudgalya/Datasets/tiny_imagenet/test/"
    data_files = os.listdir(data_folder)
    for i, image_file_name in tqdm(enumerate(data_files)):
        image_file_path = str(os.path.join(data_folder, image_file_name))
        # op = write_png("test_005/{}.png".format(i), x_hat)
        x_h, arrays, inf_bpp = sess.run([x_hat, tensors, eval_bpp], feed_dict={index: image_file_path})
        plt.imsave(os.path.join(output_folder, image_file_name), x_h[0]/255.)
        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)

        bpp.append(inf_bpp)
        full_bpp.append(len(packed.string) * 8 / (64 * 64))
        compressed_imgs.append(packed.string)

        # sess.run(op, feed_dict={index: image_file_path})

  np.save("{}/bpp.npy".format(output_folder), bpp)
  np.save("{}/full_bpp.npy".format(output_folder), full_bpp)
  pickle.dump(compressed_imgs, open("{}/compressed_imgs.p".format(output_folder), "wb"))


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="tiny_lam_0.05",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=32,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.075, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress_cifar",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  compress_cmd1 = subparsers.add_parser(
      "compress_tiny",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  # for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
  #   cmd.add_argument(
  #       "input_file",
  #       help="Input filename.")
  #   cmd.add_argument(
  #       "output_file", nargs="?",
  #       help="Output filename (optional). If not provided, appends '{}' to "
  #            "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress_cifar":
    # if not args.output_file:
    #   args.output_file = args.input_file + ".tfci"
    compress_cifar(args)
  elif args.command == "compress_tiny":
    compress_tiny(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
