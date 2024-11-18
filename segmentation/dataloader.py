"""
* dataloader.py for loading and preparing the data during Training, Validating, and Testing.

"""
import tensorflow as tf
from utils import AUTOTUNE,BUFFER_SIZE,BATCH_SIZE, train_dir,val_dir,test_dir,orig_train_dir,squeeze_mask



originl_train_dataset = tf.data.Dataset.load(orig_train_dir)
train_dataset = tf.data.Dataset.load(train_dir)
val_ds = tf.data.Dataset.load(val_dir)
#test_ds = tf.data.Dataset.load(test_dir)

single_batch = (
    originl_train_dataset
    .shuffle(BUFFER_SIZE)
    .take(1)
    .cache()
    .batch(1)
    .map(squeeze_mask)
    .prefetch(buffer_size=AUTOTUNE))
        
orig_train_batches = (
    originl_train_dataset
    .cache()
    .shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    .batch(BATCH_SIZE,num_parallel_calls=AUTOTUNE)
    .map(squeeze_mask)
    .prefetch(buffer_size=AUTOTUNE))

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    .batch(BATCH_SIZE,num_parallel_calls=AUTOTUNE)
    .map(squeeze_mask)
    .prefetch(buffer_size=AUTOTUNE))

vald_batches = val_ds.batch(BATCH_SIZE).map(squeeze_mask)

#test_batches = test_ds.batch(BATCH_SIZE).map(squeeze_mask)
