# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts Hair data to TFRecord file format with Example protos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import random
import sys
import build_data
from tqdm import tqdm
from glob import glob
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "image_folder",
    "/content/CelebAMask-HQ/CelebA-HQ-img",
    "Folder containing images",
)

tf.app.flags.DEFINE_string(
    "label_folder",
    "/content/CelebAMask-HQ/CelebAMask-HQ-mask-anno",
    "Folder containing annotations for images",
)

tf.app.flags.DEFINE_string(
    "data_split_folder",
    "/content/splits",
    "Path to folder containing data split files(train.txt, val.txt, test.txt",
)

tf.app.flags.DEFINE_string(
    "no_hair_file",
    "/content/no_hair.png",
    "Path to no hair image file",
)

tf.app.flags.DEFINE_string(
    "output_dir",
    "/content/dataset",
    "Path to save converted tfrecord of Tensorflow example",
)


def _convert_dataset():
    """Converts the Hair dataset into into tfrecord format.

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    data_split_files = tf.gfile.Glob(os.path.join(FLAGS.data_split_folder, "*.txt"))
    anno_paths = glob(os.path.join(FLAGS.label_folder, "**/*_hair.png"))
    anno_files = [os.path.basename(f) for f in anno_paths]
    
    for data_split_file in data_split_files:
        with open(data_split_file) as f:
            split_files = f.read().splitlines()
        data_output_dir = os.path.splitext(os.path.basename(data_split_file))[0]
        os.makedirs(os.path.join(FLAGS.output_dir, data_output_dir), exist_ok=True)
        
        img_names = []
        seg_names = []
        num_no_hair = 0
        for f in split_files:
            img_names.append(os.path.join(FLAGS.image_folder, f))
            # get the filename without the extension
            basename = os.path.splitext(os.path.basename(f))[0]
            basename = "{:05d}".format(int(basename))

            seg_filename = basename + "_hair.png"
            try:
                anno_index = anno_files.index(seg_filename)
                seg_names.append(anno_paths[anno_index])
            except:
                seg_names.append(FLAGS.no_hair_file)
                num_no_hair += 1
                print(f, ", ", "No hair label found")

        num_images = len(img_names)
        print("{}: {}/{}".format(data_output_dir, num_no_hair, num_images))

        image_reader = build_data.ImageReader("jpeg", channels=3)
        label_reader = build_data.ImageReader("png", channels=1)

        for i in tqdm(range(num_images)):
            output_filename = os.path.join(
                FLAGS.output_dir,
                data_output_dir,
                "{}.tfrecord".format(
                    os.path.splitext(os.path.basename(img_names[i]))[0]
                ),
            )
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                # Read the image.
                image_filename = img_names[i]
                image_data = tf.gfile.FastGFile(image_filename, "rb").read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_filename = seg_names[i]
                seg_data = tf.gfile.FastGFile(seg_filename, "rb").read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError("Shape mismatched between image and label.")
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_names[i], height, width, seg_data
                )
                tfrecord_writer.write(example.SerializeToString())


def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)
    _convert_dataset()


if __name__ == "__main__":
    tf.app.run()
