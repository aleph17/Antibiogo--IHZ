"""
* modelclass.py contains The modified training and testing steps of tf.keras.Model class.
* we are only interested on specific metrics, and we focused mainly on them.

"""
import tensorflow as tf
import numpy as np
from scipy.optimize import linear_sum_assignment
tf.config.run_functions_eagerly(True)

class CustomModel(tf.keras.Model):
  def hungarian_loss(self, y_true, y_pred):
    def compute_batch_loss(args):
      y_true, y_pred = args
      # Compute pairwise Euclidean distance matrix
      dist_matrix = tf.norm(
        y_true[:, tf.newaxis] - y_pred[tf.newaxis, :],
        axis=-1
      )
      # Convert to numpy for Hungarian algorithm
      dist_matrix_np = dist_matrix.numpy()
      # Apply Hungarian algorithm
      row_ind, col_ind = linear_sum_assignment(dist_matrix_np)
      # Calculate loss for this batch
      return tf.reduce_sum(tf.gather_nd(
        dist_matrix,
        tf.stack([row_ind, col_ind], axis=1)
      ))

    # Map the computation over the batch dimension
    batch_losses = tf.map_fn(
      compute_batch_loss,
      (y_true, y_pred),
      fn_output_signature=tf.float32
    )

    # Return average loss across batch
    return tf.reduce_mean(batch_losses)


  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")

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
        loss_value = self.hungarian_loss(y_true = target, y_pred = predictions)
    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    # print(loss_value)
    grads = tape.gradient(loss_value, trainable_vars)
    # print(grads)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = self.hungarian_loss(y_true=target, y_pred=predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result()}

  @property
  def metrics(self):
      return [self.loss_tracker]
