
from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv
import tensorflow as tf


# root_path = getenv("ROOT_DIR")
root_path = '/mloscratch/sayfiddi/segmentation'
img_pth = path.join(root_path,"base_data/images")
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
BUFFER_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
# The required image size.


IMG_SIZE = 1024
OUTPUT_CLASSES = 2
EXPR_BATCHES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128, 256, 512]
EXPR_FILTERS = [8, 12, 32, 64]
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

  
# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

