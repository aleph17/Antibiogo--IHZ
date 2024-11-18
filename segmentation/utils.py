
from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv
import tensorflow as tf
import cv2 as cv
import numpy as np


# root_path = getenv("ROOT_DIR")
root_path = '/home/muhammad-ali/working'
img_pth = path.join(root_path,"base_data/padded")
train_dir = path.join(root_path,"tf_record/Train")
val_dir = path.join(root_path,"tf_record/Valid")
test_dir = path.join(root_path,"tf_record/Test")
orig_train_dir = path.join(root_path,"tf_record/Original_Train")


tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
initial_bias = -1.84606594


# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1024
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
# The required image size.


IMG_SIZE = 256
OUTPUT_CLASSES = 2
EXPR_BATCHES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128, 256, 512]
EXPR_FILTERS = [8, 16, 32, 64]
EXPR_WEIGHTS = [0.1, 0.01, 0.001, 0.0001]


def display(display_list:List)->None:
  """
  [true_image,true_mask,predicted_mask] -> display
  """
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for idx in range(len(display_list)):
    plt.subplot(1, len(display_list), idx+1)
    plt.title(title[idx])
    plt.imshow(array_to_img(display_list[idx]))
    plt.axis('off')
  plt.show()


def squeeze_mask(img,mask):
        return img, tf.squeeze(mask, axis = -1)
#  tf.squeeze(mask,axis=-1)  changed

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def calculate_iou(pred_xyr: list, truth_xyr: list, size: int):
    pred = np.zeros([size, size])
    truth = np.zeros([size, size])
    for circ in pred_xyr:
        x, y, r = circ
        cv.circle(pred, (int(x),int(y)), int(r), 1, -1)
    for circ in truth:
        x, y, r = circ
        cv.circle(truth, (int(x),int(y)), int(r), 1, -1)
    
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    
    if union == 0:
        return 1 if intersection == 0 else 0  # Perfect match if both are empty, else 0
    return intersection / union
  
# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

