from numpy.random import seed as seednp
import tensorflow as tf
from typing import Tuple
from pathlib import Path
from os import listdir
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir

tf.random.set_seed(tf_global_seed)
seednp(np_seed)

vald_ratio = 0.2
test_ratio = 0.1

img_dir = Path(img_pth).with_suffix('')

img_list_ds = tf.data.Dataset.list_files([str(img_dir / '*.jp*')], shuffle=False)

data_count = tf.data.experimental.cardinality(img_list_ds).numpy()


def normalize(input_img: tf.Tensor) -> tf.Tensor:
    input_image = -1 + tf.cast(input_img, tf.float32) / 127.5
    return input_image


def load_image(input_img: str, input_mask: str) -> tuple[tf.Tensor, tf.Tensor]:
    input_img = tf.io.decode_jpeg(input_img)
    input_mask = tf.io.decode_jpeg(input_mask, channels = 1)

    input_img = normalize(input_img)
    input_mask = tf.cast(input_mask / 255, tf.int8)

    r, g, b = input_img[:, :, 0], input_img[:, :, 1], input_img[:, :, 2]
    input_img = tf.stack([r, g, b], axis=-1)

    return input_img, input_mask


def process_path(image_path: str) -> Tuple:
    mask_path = tf.strings.regex_replace(image_path, "images", "masks_01")
    img = tf.io.read_file(image_path)
    mask = tf.io.read_file(mask_path)
    img, mask = load_image(img, mask)
    return img, mask


ready_ds = img_list_ds.take(data_count).shuffle(BUFFER_SIZE, seed=shuffle_data_seed).map(process_path,
                                                                                         num_parallel_calls=AUTOTUNE)

test_size = int(data_count * test_ratio)
test_ds = ready_ds.take(test_size)
val_size = int(data_count * vald_ratio)
val_ds = ready_ds.skip(test_size).take(val_size)
train_size = data_count - val_size - test_size
train_ds_original = ready_ds.skip(test_size + val_size).take(train_size)

Path(orig_train_dir).mkdir(parents=True)
dir_files = listdir(orig_train_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

train_ds_original.save(orig_train_dir)

Path(val_dir).mkdir(parents=True)
dir_files = listdir(val_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

val_ds.save(val_dir)

Path(test_dir).mkdir(parents=True)
dir_files = listdir(test_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

test_ds.save(test_dir)
