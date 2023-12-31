import random
import shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tqdm import tqdm


import os, sys


class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def mask_by_color(img: tf.Tensor, col: Tuple[int, int, int]) -> tf.Tensor:
    img = tf.cast(img == col, dtype=tf.uint8)
    img = tf.reduce_sum(img, axis=-1) == 3
    return tf.cast(img, dtype=tf.float32)


def masking(
        img: tf.Tensor,
        class_colors: List[Tuple[int, int, int]],
        stack_axis=-1
) -> tf.Tensor:
    img = tf.image.convert_image_dtype(img, tf.uint8, saturate=True)
    img = tf.stack([
        mask_by_color(img, col)
        for col in class_colors
    ], axis=stack_axis)

    return img


def rgb_to_onehot(rgb_arr):
    color_dict = {
        0: (0, 0, 0),
        1: (255, 255, 255)
    }
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_dict[i],
                              axis=1).reshape(shape[:2])
    return arr


def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in color_dict.keys():
        output[single_layer == k] = color_dict[k]
    return np.uint8(output)


def sample_more_likely_in_the_middle(range_length):
    epsilon = 0.000001
    percentage = np.clip(
        random.betavariate(5, 5),
        a_min=0,
        a_max=1 - epsilon
    )
    return int(percentage * range_length)


def random_slice(np_img, width, height=None):
    if height is None:
        height = width

    mask_width, mask_height = np_img.shape[:2]

    x_min = sample_more_likely_in_the_middle(mask_width - width)
    y_min = sample_more_likely_in_the_middle(mask_height - height)
    # x_min = np.clip(x_min, 0, mask_width - width)
    # y_min = np.clip(y_min, 0, mask_height - height)

    x_max = x_min + width
    y_max = y_min + height

    return np.s_[y_min: y_max, x_min: x_max, ...]


def combine_binary_masks(mask_1, mask_2):
    tf.assert_equal(tf.reduce_sum(mask_1),
                    mask_1.shape[0] * mask_1.shape[1] * 1.,
                    message="first assertion")
    tf.assert_equal(tf.reduce_sum(mask_2),
                    mask_2.shape[0] * mask_2.shape[1] * 1.,
                    message="second assertion")

    foreground = mask_1[..., 1] + mask_2[..., 1]
    foreground = tf.clip_by_value(foreground, 0, 1)

    ones = tf.ones_like(mask_1[..., 0])
    background = ones - foreground

    result = tf.stack((background, foreground), axis=-1)
    tf.assert_equal(tf.reduce_sum(result),
                    result.shape[0] * result.shape[1] * 1.,
                    message="thirdasserstion")
    return result



def validate_images(path: Path):
    print("Validating files in", path)
    for path in tqdm(list(path.rglob("*.*"))):
        try:
            img = tf.io.read_file(str(path))
            # convert the compressed string to a 3D uint8 tensor
            tf.image.decode_image(img)
        except:
            print("could not open", path)


def copy_to_folder(src: Path, target):
    target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(src, target)
