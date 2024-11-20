"""
* callback.py contains callsbacks that are needed for saving the models and predictions during training.

"""
import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from dataloader import single_batch
import wandb
from utils import create_mask


for images, masks in single_batch:
    sample_image, sample_mask = images[0], masks[0]

# Callbacks 
#EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=3)
checkpoint_filepath = 'BaseModel/checkpoint.model.keras'
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min'
                                                        )


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs=None):
      wandb.log({"Prediction": [wandb.Image(array_to_img(sample_image), caption="Input Image"),
                               wandb.Image(array_to_img(sample_mask[...,tf.newaxis]), caption="True Mask"),
                               wandb.Image(array_to_img(create_mask(self.model.predict(sample_image[tf.newaxis, ...]))),
                                           caption="Predicted Mask start_of_training")]})
    
  def on_epoch_end(self, epoch, logs=None):
      if epoch%10==0:
          wandb.log({"Prediction": [wandb.Image(array_to_img(sample_image), caption="Input Image"),
                                   wandb.Image(array_to_img(sample_mask[...,tf.newaxis]), caption="True Mask"),
                                   wandb.Image(array_to_img(create_mask(self.model.predict(sample_image[tf.newaxis, ...]))),
                                               caption="Predicted Mask epoch:{}".format(epoch))]})


          