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

  def compute_prediction_ratio(self, y_true, y_pred):
    def count_nonzero_vectors(tensor):
        # Consider a vector non-zero if any of its components are non-zero
        return tf.reduce_sum(tf.cast(
            tf.reduce_any(tf.not_equal(tensor, 0), axis=-1),
            tf.float32
        ))
    
    # Count non-zero vectors in both tensors
    true_count = count_nonzero_vectors(y_true)
    pred_count = count_nonzero_vectors(y_pred)
    
    # Compute ratio (pred_count / true_count)
    # Add small epsilon to avoid division by zero
    ratio = pred_count / (true_count + tf.keras.backend.epsilon())
    
    return ratio


  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.pred_ratio_tracker = tf.keras.metrics.Mean(name="pred_ratio")

  def train_step(self, data):
    image, target = data
    with tf.GradientTape() as tape:
        predictions = self(image, training=True)
        loss_value = self.hungarian_loss(y_true = target, y_pred = predictions)

    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))

    pred_ratio = self.compute_prediction_ratio(target, predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.pred_ratio_tracker.update_state(pred_ratio)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "pred_ratio": self.pred_ratio_tracker.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = self.hungarian_loss(y_true=target, y_pred=predictions)
    pred_ratio = self.compute_prediction_ratio(target, predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.pred_ratio_tracker.update_state(pred_ratio)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "pred_ratio": self.pred_ratio_tracker.result()}

  @property
  def metrics(self):
      return [self.loss_tracker, self.pred_ratio_tracker]
