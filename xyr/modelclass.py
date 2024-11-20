# """
# * modelclass.py contains The modified training and testing steps of tf.keras.Model class.
# * we are only interested on specific metrics, and we focused mainly on them.

# """
# import tensorflow as tf
# from utils import BATCH_SIZE, IMG_SIZE
# from tensorflow.experimental import numpy as tnp

# tf.config.run_functions_eagerly(True)

# class CustomModel(tf.keras.Model):
#   def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     self.loss_tracker = tf.keras.metrics.Mean(name="loss")
#     self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
#     self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
#     self.iou_metric = tf.keras.metrics.Mean(name="iou")
  
#   def compute_circle_iou(self, pred_circles, true_circles):
#     def circles_to_boxes(circles):
#         x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
#         return tf.stack([
#             x - r,   # x_min
#             y - r,   # y_min
#             x + r,   # x_max
#             y + r    # y_max
#         ], axis=-1)
    
#     # Convert to boxes
#     pred_boxes = circles_to_boxes(pred_circles)
#     true_boxes = circles_to_boxes(true_circles)
    
#     # Compute intersections
#     x1 = tf.maximum(pred_boxes[:, None, 0], true_boxes[None, :, 0])
#     y1 = tf.maximum(pred_boxes[:, None, 1], true_boxes[None, :, 1])
#     x2 = tf.minimum(pred_boxes[:, None, 2], true_boxes[None, :, 2])
#     y2 = tf.minimum(pred_boxes[:, None, 3], true_boxes[None, :, 3])
    
#     # Compute intersection areas
#     intersect_w = tf.maximum(0.0, x2 - x1)
#     intersect_h = tf.maximum(0.0, y2 - y1)
#     intersection = intersect_w * intersect_h
    
#     # Compute box areas
#     pred_area = (pred_boxes[:, None, 2] - pred_boxes[:, None, 0]) * \
#                 (pred_boxes[:, None, 3] - pred_boxes[:, None, 1])
#     true_area = (true_boxes[None, :, 2] - true_boxes[None, :, 0]) * \
#                 (true_boxes[None, :, 3] - true_boxes[None, :, 1])
    
#     # Compute union
#     union = pred_area + true_area - intersection
    
#     # Compute IoU
#     iou = intersection / (union + 1e-8)
#     return iou

#   def train_step(self, data):
#     image, target = data
#     # image = tf.ensure_shape(image, (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))
#     # target = tf.ensure_shape(target, (BATCH_SIZE, 3))
#     # Open a GradientTape.
#     with tf.GradientTape() as tape:
#         # Forward pass.
#         predictions = self(image, training=True)
#         # print("Predictions shape:", predictions.shape)
#         # Compute the loss.
#         iou_values = self.compute_circle_iou(predictions, target)
#         # print('iou:', iou_values)
#         loss_value =  tf.convert_to_tensor(1 - tf.reduce_mean(iou_values), dtype=tf.float32)
#         # print('loss:',loss_value)

#     # Compute gradients and update weights
#     trainable_vars = self.trainable_variables
#     # print("Trainable Variables:", trainable_vars)
#     grads = tape.gradient(loss_value, trainable_vars)
#     # print("Gradients:", grads)
#     self.optimizer.apply_gradients(zip(grads, trainable_vars))
#     # Update metrics
#     self.loss_tracker.update_state(loss_value)
#     self.mae_metric.update_state(target, predictions)
#     self.mse_metric.update_state(target, predictions)
#     self.iou_metric.update_state(iou_values)
#     # Return a dict mapping metric names to current value
#     return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result(), "iou": self.iou_metric.result()}
  
#   def test_step(self, data):
#     # Unpack the data.
#     image, target = data
#     # Compute predictions
#     predictions = self(image, training=False)
#     # Compute the loss.
#     iou_values = self.compute_circle_iou(predictions, target)
#     loss_value = 1 - tf.reduce_mean(iou_values)
#     # Update metrics
#     self.loss_tracker.update_state(loss_value)
#     self.mae_metric.update_state(target, predictions)
#     self.mse_metric.update_state(target, predictions)
#     self.iou_metric.update_state(iou_values)
#     # Return a dict mapping metric names to current value
#     return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result(), "iou": self.iou_metric.result()}

#   @property
#   def metrics(self):
#       return [self.loss_tracker, self.mae_metric, self.mse_metric, self.iou_metric]


##   keras IoU
# keras_cv.losses.IoULoss(bounding_box_format = 'center_xywh')(y_true, y_pred)
# def converter(targets):
#     x,y,r = targets[..., 0], targets[..., 1], targets[..., 2] 
#     return tf.stack([x, y,2 * r,2 * r], axis=-1)

## better loss for xyr
# def crcl(targets):
#   x,y,r = targets[..., 0], targets[..., 1], targets[..., 2] 
#   r = sqrpi*r
#   return tf.stack([x, y,2 * r,2 * r], axis=-1)


#############   ONLY TO TRAIN WITH MSE    #############


import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.experimental import numpy as tnp
sqrpi = tf.sqrt(tnp.pi)

def area(targets):
  x,y,r = targets[..., 0], targets[..., 1], targets[..., 2] 
  r = sqrpi*r
  return tf.stack([x, y,2 * r,2 * r], axis=-1)

class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

  def train_step(self, data):
    # print(data)
    image, target = data
    # print("Train step - Image shape:", tf.shape(image))
    # print("Train step - Target shape:", tf.shape(target))
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = self(image, training=True)
        # print("Predictions shape:", predictions.shape)
        # Compute the loss.
        loss_value = tf.keras.losses.MeanSquaredError()(y_true=area(target), y_pred=area(predictions))
    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(target, predictions)
    self.mse_metric.update_state(target, predictions)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = tf.keras.losses.MeanSquaredError()(y_true=target, y_pred=predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(target, predictions)
    self.mse_metric.update_state(target, predictions)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  @property
  def metrics(self):
      return [self.loss_tracker, self.mae_metric, self.mse_metric]
