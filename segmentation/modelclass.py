"""
* modelclass.py contains The modified training and testing steps of tf.keras.Model class.
* we are only interested on specific metrics, and we focused mainly on them.

"""
import tensorflow as tf

class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.meaniou_metric = tf.keras.metrics.MeanIoU(num_classes=2,name="iou")
    self.acc_metric = tf.keras.metrics.Accuracy(name="accuracy")

  def train_step(self, data):
    # Unpack the data.
    image, mask = data
    # Open a GradientTape.
    with tf.GradientTape() as tape:
      # Forward pass.
      logits = self(image,training=True)
      # Compute the loss.
      loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=mask, y_pred=logits)
    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.meaniou_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    self.acc_metric.update_state(mask, tf.math.argmax(logits, axis=-1))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "iou": self.meaniou_metric.result(),
            "accuracy":self.acc_metric.result()}
  
  def test_step(self, data):
    # Unpack the data.
    image, mask = data
    # Compute predictions
    pred_mask = self(image, training=False)
    # Compute the loss.
    loss_value = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true=mask, y_pred=pred_mask)
    # Update the metrics.
    self.loss_tracker.update_state(loss_value)
    self.meaniou_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    self.acc_metric.update_state(mask, tf.math.argmax(pred_mask, axis=-1))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "iou": self.meaniou_metric.result(),
            "accuracy": self.acc_metric.result()}

  @property
  def metrics(self):
    return [self.loss_tracker, self.meaniou_metric,self.acc_metric]
