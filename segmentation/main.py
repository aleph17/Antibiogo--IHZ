# from dataloader import orig_train_batches, vald_batches, single_batch
# from basemodel import model_net_MobileNetv2, filters
from callback import DisplayCallback  # ,EarlyStopping_callback, savemodel_callback
import wandb

# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import optimAdam, EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS
# ----------------------
import sys
from tensorflow_examples.models.pix2pix import pix2pix
from ssl import _create_default_https_context, _create_unverified_context
from utils import IMG_SIZE, OUTPUT_CLASSES, initial_bias, orig_train_dir, BUFFER_SIZE, AUTOTUNE, squeeze_mask, val_dir, \
    test_dir, root_path, train_dir

import tensorflow as tf
from modelclass import CustomModel

# ------------------------
if __name__ == "__main__":
    filters = int(sys.argv[1])
    batch = int(sys.argv[2])
    weight = float(sys.argv[3])

    # ------------------
    def unet_model(output_channels: int):
        _create_default_https_context = _create_unverified_context
        base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # (Batch_size x Height x Width x No. channels): (None x 128 x 128 x 96)
            'block_3_expand_relu',  # (None x 64 x 64 x 144)
            'block_6_expand_relu',  # (None x 32 x 32 x 192)
            'block_13_expand_relu',  # (None x 16 x 16 x 576)
            'block_16_project',  # (None x 8 x 8 x 960) -> (None x 8 x 8 x 320)
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(int(filters), 3),  # 8x8 -> 16x16
            pix2pix.upsample(int(filters / 2), 3),  # 16x16 -> 32x32
            pix2pix.upsample(int(filters / 4), 3),  # 32x32 -> 64x64
            pix2pix.upsample(int(filters / 8), 3),  # 64x64 -> 128x128
        ]
        inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # Initialize the bias of the last layer properly to speed up convergence.(for the case of Imbalanced dataset)
        output_bias = tf.keras.initializers.Constant(value=initial_bias)

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding='same',
            bias_initializer=output_bias)  # 128x128 -> 256x256

        x = last(x)
        return CustomModel(inputs=inputs, outputs=x)


    model_net_MobileNetv2 = unet_model(output_channels=OUTPUT_CLASSES)
    optimAdam = tf.keras.optimizers.Adam(learning_rate=0.0003, weight_decay = weight)
    model_net_MobileNetv2.compile(optimizer=optimAdam)

    originl_train_dataset = tf.data.Dataset.load(orig_train_dir)
    orig_train_batches = (originl_train_dataset.cache().shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(batch,num_parallel_calls=AUTOTUNE).map(squeeze_mask).prefetch(buffer_size=AUTOTUNE))
    
    val_ds = tf.data.Dataset.load(val_dir)
    vald_batches = val_ds.batch(batch).map(squeeze_mask)

    train_dataset = tf.data.Dataset.load(train_dir)
    train_batches = (train_dataset.cache().shuffle(BUFFER_SIZE,reshuffle_each_iteration=True).batch(batch,num_parallel_calls=AUTOTUNE).map(squeeze_mask).prefetch(buffer_size=AUTOTUNE))

    # ------------------
    # with open(f"{root_path}/ModelSummary/model_{filters}.txt", "w") as fily:
    #     model_net_MobileNetv2.summary(print_fn=lambda x: fily.write(x + "\n"))
    model = model_net_MobileNetv2
    EPOCHS = 1000
    # Launch an experiment
    wandb.init(project="segmentation",
        name=f"F{filters:<3} B{batch:<2} W{weight:<6} | {date.today()}", config={"epoch": EPOCHS},)

    config = wandb.config
    # Add WandbMetricsLogger to log metrics
    wandb_callbacks = WandbMetricsLogger()
    model_history = model.fit(orig_train_batches,
                              epochs=config.epoch,
                              verbose=0,
                              validation_data=vald_batches,
                              callbacks=[wandb_callbacks, DisplayCallback()]
                              # +EarlyStopping_callback
                              )
    # Mark the run as finished
    model.save(f'/mloscratch/sayfiddi/segmentation/SavedModels/modelF{filters}B{batch}W{weight}.h5')
    wandb.finish()