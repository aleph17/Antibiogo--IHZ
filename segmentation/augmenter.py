from numpy.random import seed as seednp
import tensorflow as tf
import tensorflow_io as tfio
from typing import Tuple
from pathlib import Path
from scipy.ndimage import rotate as ndimage_rotate
from os import listdir
from utils import IMG_SIZE,AUTOTUNE,tf_global_seed,np_seed,train_dir,orig_train_dir


# The global random seed.
tf.random.set_seed(tf_global_seed)
seednp(np_seed)

train_ds_original = tf.data.Dataset.load(orig_train_dir)
counter = tf.data.Dataset.counter()
train_ds = tf.data.Dataset.zip((train_ds_original, (counter,counter)))


@tf.py_function(Tout=tf.uint8)
def random_rotate_mask(mask, rotation_angle):
    return ndimage_rotate(mask, rotation_angle, reshape=False, mode='nearest')


def tf_random_rotate_mask(mask, rotation_angle):
    mas_shape = mask.shape
    mask = random_rotate_mask(mask, rotation_angle)
    mask.set_shape(mas_shape)
    return mask


@tf.py_function(Tout=tf.float32)
def random_rotate_image(image, rotation_angle):
    return ndimage_rotate(image, rotation_angle, reshape=False, mode='nearest')


def tf_random_rotate_image(image, rotation_angle):
    img_shape = image.shape
    image = random_rotate_image(image, rotation_angle)
    image.set_shape(img_shape)
    return image



class RandomContrast(tf.keras.Layer):
    def __init__(self, lower_value: float = 0.1, upper_value: float = 0.9):
        super().__init__()
        self.lower = lower_value
        self.upper = upper_value

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_contrast(img, lower=self.lower, upper=self.upper, seed=new_seed), mask


# 1. Adjust the hue of RGB image.
class AdjustHue(tf.keras.Layer):
    def __init__(self, delta_value: float = 0.25):
        super().__init__()
        self.delta = delta_value
        if delta_value < 0 or delta_value > 0.5:
            raise ValueError("Delta value should be in the range [0,0.5], not %s" % delta_value)

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_hue(img, max_delta=self.delta, seed=new_seed), mask


# 2. Radomize jpeg encoding quality.
class RandomizeJpeg(tf.keras.Layer):
    def __init__(self, min_quality: int = 0, max_quality: int = 100):
        super().__init__()
        self.min_quality = min_quality
        self.max_quality = max_quality

        if min_quality < 0 or max_quality < 0 or min_quality >= max_quality:
            raise ValueError("min_qulaity and max_quality should be in the range [0,100] and min_quality<max_quality.")

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_jpeg_quality(img,
                                                      min_jpeg_quality=self.min_quality,
                                                      max_jpeg_quality=self.max_quality,
                                                      seed=new_seed), mask

        # return tf.map_fn(lambda sing_img: tf.image.stateless_random_jpeg_quality(sing_img,
        #                                                                                    min_jpeg_quality=self.min_quality,
        #                                                                                   max_jpeg_quality=self.max_quality,
        #                                                                                  seed=new_seed),img),mask


# 3. Adjust the saturation of RGB images.
class ImageSaturation(tf.keras.Layer):
    def __init__(self, lower_sat_f: float = 0.1, upper_sat_f: float = 5):
        super().__init__()
        self.upper = upper_sat_f
        self.lower = lower_sat_f

        if lower_sat_f < 0 or upper_sat_f <= lower_sat_f:
            raise ValueError(
                "Lower saturation factor should be greater than or equal to zero, and less than the Upper Saturation factor.")

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_saturation(img, lower=self.lower, upper=self.upper, seed=new_seed), mask


# 4. Change the brightness of an image.
class ImageBrightness(tf.keras.Layer):
    def __init__(self, brightness_value: float = 0.95):
        super().__init__()
        self.MAX_delta = brightness_value

        if brightness_value < 0:
            raise ValueError("The brightness factor should not be negative.")

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        # Make a new seed.
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_brightness(img, max_delta=self.MAX_delta,
                                                    seed=new_seed), mask


### The remaining cases will augment Image & Mask.###
# 5. Random horizontal flip.
class HorizontalFlip(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_flip_left_right(img, new_seed), tf.image.stateless_random_flip_left_right(mask,
                                                                                                                   new_seed)

# 6. Random vertical flip.


class VerticalFlip(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        new_seed = tf.random.split(SEED, num=1)[0, :]

        return tf.image.stateless_random_flip_up_down(img, new_seed), tf.image.stateless_random_flip_up_down(mask,
                                                                                                             new_seed)


# 7. Random Rotation.
class RandomRotation(tf.keras.Layer):
    def __init__(self, rotation_angle: float = 30):
        super().__init__()
        self.rotation_angle = rotation_angle

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask

        return tf_random_rotate_image(img, self.rotation_angle), tf.cast(
            tf_random_rotate_image(mask, self.rotation_angle),
            tf.uint8)  # tf.map_fn(lambda single_img: tf_random_rotate_image(single_img,self.rotation_angle),img), tf.map_fn(lambda single_mask: tf_random_rotate_mask(single_mask,self.rotation_angle),mask)


# 8. Random Crop.
class RandomCrop(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        new_seed = tf.random.split(SEED, num=1)[0, :]
        # Double the size of the image and mask to crop the desired size [IMG_SIZE,IMG_SIZE].
        img = tf.image.resize(img, (IMG_SIZE * 2, IMG_SIZE * 2))
        mask = tf.image.resize(mask, (IMG_SIZE * 2, IMG_SIZE * 2),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.image.stateless_random_crop(img, size=[IMG_SIZE, IMG_SIZE, 3],
                                              seed=new_seed), tf.image.stateless_random_crop(mask,
                                                                                             size=[IMG_SIZE, IMG_SIZE,
                                                                                                   1], seed=new_seed)
        # return tf.map_fn(lambda single_img: tf.image.stateless_random_crop(single_img,size=[IMG_SIZE,IMG_SIZE,3],
        #                                                                  seed=new_seed),img),tf.map_fn(lambda single_mask: tf.image.stateless_random_crop(single_mask,
        #                                                                                                                                                  size=[IMG_SIZE,IMG_SIZE,1],
        #                                                                                                                                                 seed=new_seed),mask)


## Augmentations that work on Image only.
# 9. Convert RGB to BGR
class RgbToBgr(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        return tfio.experimental.color.rgb_to_bgr(img), mask


# 10. Convert RGB to CIE XYZ
class RgbToCieXyz(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        return tfio.experimental.color.rgb_to_xyz(img), mask


# 11. Convert a RGB image to HSV.
class RgbToHsv(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        return tfio.experimental.color.rgb_to_hsv(img), mask


# 12. Convert a RGB image to CIE LAB.
class RgbToCieLab(tf.keras.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_mask: Tuple, SEED: Tuple) -> Tuple:
        img, mask = img_mask
        return tfio.experimental.color.rgb_to_lab(img), mask


# Augment the data:
# 0
cotras_data = train_ds.map(RandomContrast(), num_parallel_calls=AUTOTUNE)
cotras_data28 = train_ds.map(RandomContrast(0.2, 0.8), num_parallel_calls=AUTOTUNE)
cotras_data37 = train_ds.map(RandomContrast(0.3, 0.7), num_parallel_calls=AUTOTUNE)
cotras_data1131 = train_ds.map(RandomContrast(1.1, 3.1), num_parallel_calls=AUTOTUNE)

# 1
adjhue_data1 = train_ds.map(AdjustHue(0.1), num_parallel_calls=AUTOTUNE)
adjhue_data025 = train_ds.map(AdjustHue(), num_parallel_calls=AUTOTUNE)
adjhue_data5 = train_ds.map(AdjustHue(0.5), num_parallel_calls=AUTOTUNE)
adjhue_data2 = train_ds.map(AdjustHue(0.2), num_parallel_calls=AUTOTUNE)

# 2
Randjpeg_data = train_ds.map(RandomizeJpeg(), num_parallel_calls=AUTOTUNE)
Randjpeg_data1090 = train_ds.map(RandomizeJpeg(10, 90), num_parallel_calls=AUTOTUNE)
Randjpeg_data2080 = train_ds.map(RandomizeJpeg(20, 80), num_parallel_calls=AUTOTUNE)
# 3
satur_data = train_ds.map(ImageSaturation(), num_parallel_calls=AUTOTUNE)
satur_data10 = train_ds.map(ImageSaturation(0.1, 10), num_parallel_calls=AUTOTUNE)
# 4
bright_data95 = train_ds.map(ImageBrightness(), num_parallel_calls=AUTOTUNE)
bright_data9 = train_ds.map(ImageBrightness(0.9), num_parallel_calls=AUTOTUNE)
bright_data8 = train_ds.map(ImageBrightness(0.8), num_parallel_calls=AUTOTUNE)
bright_data7 = train_ds.map(ImageBrightness(0.7), num_parallel_calls=AUTOTUNE)
bright_data6 = train_ds.map(ImageBrightness(0.6), num_parallel_calls=AUTOTUNE)
bright_data5 = train_ds.map(ImageBrightness(0.5), num_parallel_calls=AUTOTUNE)
bright_data4 = train_ds.map(ImageBrightness(0.4), num_parallel_calls=AUTOTUNE)
bright_data3 = train_ds.map(ImageBrightness(0.3), num_parallel_calls=AUTOTUNE)
bright_data2 = train_ds.map(ImageBrightness(0.2), num_parallel_calls=AUTOTUNE)
# 5
horizontal_data = train_ds.map(HorizontalFlip(), num_parallel_calls=AUTOTUNE)
# 6
vertical_data = train_ds.map(VerticalFlip(), num_parallel_calls=AUTOTUNE)
# 7
randomrotation_data30 = train_ds.map(RandomRotation(), num_parallel_calls=AUTOTUNE)
randomrotation_data60 = train_ds.map(RandomRotation(60), num_parallel_calls=AUTOTUNE)
randomrotation_data90 = train_ds.map(RandomRotation(90), num_parallel_calls=AUTOTUNE)
randomrotation_data120 = train_ds.map(RandomRotation(120), num_parallel_calls=AUTOTUNE)
randomrotation_data150 = train_ds.map(RandomRotation(150), num_parallel_calls=AUTOTUNE)
randomrotation_data_30 = train_ds.map(RandomRotation(-30), num_parallel_calls=AUTOTUNE)
randomrotation_data_60 = train_ds.map(RandomRotation(-60), num_parallel_calls=AUTOTUNE)
randomrotation_data_90 = train_ds.map(RandomRotation(-90), num_parallel_calls=AUTOTUNE)
randomrotation_data_120 = train_ds.map(RandomRotation(-120), num_parallel_calls=AUTOTUNE)
randomrotation_data_150 = train_ds.map(RandomRotation(-150), num_parallel_calls=AUTOTUNE)
# 8
randomcrop_data = train_ds.map(RandomCrop(), num_parallel_calls=AUTOTUNE)
# 9
bgr_data = train_ds.map(RgbToBgr(), num_parallel_calls=AUTOTUNE)
# 10
xyz_data = train_ds.map(RgbToCieXyz(), num_parallel_calls=AUTOTUNE)
# 11
hsv_data = train_ds.map(RgbToHsv(), num_parallel_calls=AUTOTUNE)
# 12
cielab_data = train_ds.map(RgbToCieLab(), num_parallel_calls=AUTOTUNE)

# The training dataset with the augmented one.
train_dataset = train_ds_original.concatenate(cotras_data).concatenate(cotras_data28).concatenate(
    cotras_data37).concatenate(cotras_data1131).concatenate(adjhue_data1).concatenate(adjhue_data025).concatenate(
    adjhue_data5)
train_dataset = train_dataset.concatenate(adjhue_data2).concatenate(Randjpeg_data).concatenate(
    Randjpeg_data1090).concatenate(Randjpeg_data2080).concatenate(satur_data).concatenate(satur_data10).concatenate(
    bright_data95).concatenate(bright_data9)
train_dataset = train_dataset.concatenate(bright_data8).concatenate(bright_data7).concatenate(bright_data6).concatenate(
    bright_data5).concatenate(bright_data4).concatenate(bright_data3).concatenate(bright_data2).concatenate(
    horizontal_data).concatenate(vertical_data)
# train_dataset = train_dataset.concatenate(randomrotation_data30).concatenate(
#     randomrotation_data60).concatenate(randomrotation_data90).concatenate(randomrotation_data120).concatenate(
#     randomrotation_data150)
# train_dataset = train_dataset.concatenate(randomrotation_data_30).concatenate(randomrotation_data_60).concatenate(
#     randomrotation_data_90).concatenate(randomrotation_data_120).concatenate(randomrotation_data_150).concatenate(
#     randomcrop_data)

""" above we have problems with the data being concatenated. -> shape (None, None, 3)"""
train_dataset = train_dataset.concatenate(bgr_data).concatenate(xyz_data).concatenate(hsv_data).concatenate(cielab_data)

# Save the Training dataset with the augmented one.
Path(train_dir).mkdir(parents=True)
dir_files = listdir(train_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The direcotry exists and is not empty.")
train_dataset.save(train_dir)
