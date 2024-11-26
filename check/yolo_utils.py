from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv
import tensorflow as tf
from PIL import Image, ImageDraw


# root_path = getenv("ROOT_DIR")
root_path = '/home/muhammad-ali/working/check'
img_pth = path.join(root_path,"base_data/ready")
annot_path = path.join(root_path, 'base_data/pellet_annot.json')
classes_path = path.join(root_path, 'base_data/classes.json')



train_dir = path.join(root_path,"yolo_tf_record/Train")
val_dir = path.join(root_path,"yolo_tf_record/Valid")
test_dir = path.join(root_path,"yolo_tf_record/Test")
orig_train_dir = path.join(root_path,"yolo_tf_record/Original_Train")

tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
initial_bias = -1.84606594


# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1024
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
GLOBAL_CLIPNORM = 10.0
class_ids = ["None", "ihz", 'pellet', 'dish']
class_mapping = dict(zip(range(len(class_ids)), class_ids))
# The required image size.


IMG_SIZE = 512
OUTPUT_CLASSES = 2


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


def drawer(image: list, tars: list):
  colors = [(0,255,0), (255, 0, 0)]
  for i in range(len(tars)):
    target = tars[i]
    for circle in target:
      x, y, w, h = circle  # center (x, y) and radius (r)
      r = (w+h)/4
      # Convert to top-left and bottom-right coordinates
      if x>10 and y>10:
        top_left = (x - r, y - r)
        bottom_right = (x + r, y + r)

        # Ensure the coordinates are in the correct order
        top_left = (min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1]))
        bottom_right = (max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1]))

        # Draw ellipse
        draw = ImageDraw.Draw(image)
        draw.ellipse([top_left, bottom_right], outline=colors[i], width=3)
  return image


def targetize(pred_target):
  pred_target = pred_target[0]
  # pred_target = pred_target.tolist()
  return pred_target



