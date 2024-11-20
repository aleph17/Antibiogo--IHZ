"""
* basemodel.py contains The baseline model architecture.
* we use MobileNetv2 as the baseline. We are following these two implementations:
    - https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
    - https://www.tensorflow.org/tutorials/images/segmentation#define_the_model

Reference:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)

"""
from tensorflow_examples.models.pix2pix import pix2pix
from ssl import _create_default_https_context, _create_unverified_context
from utils import IMG_SIZE,OUTPUT_CLASSES, initial_bias
import tensorflow as tf
from modelclass import CustomModel

filters = 8
def unet_model(output_channels:int):
    _create_default_https_context = _create_unverified_context
    base_model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # (Batch_size x Height x Width x No. channels): (None x 128 x 128 x 96)
        'block_3_expand_relu',   # (None x 64 x 64 x 144) 
        'block_6_expand_relu',   # (None x 32 x 32 x 192) 
        'block_13_expand_relu',  # (None x 16 x 16 x 576) 
        'block_16_project',      # (None x 8 x 8 x 960) -> (None x 8 x 8 x 320)
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(int(filters), 3),  # 8x8 -> 16x16
        pix2pix.upsample(int(filters/2), 3),  # 16x16 -> 32x32
        pix2pix.upsample(int(filters/4), 3),  # 32x32 -> 64x64
        pix2pix.upsample(int(filters/8), 3),   # 64x64 -> 128x128
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
        bias_initializer = output_bias)  #128x128 -> 256x256

    x = last(x)
    return CustomModel(inputs=inputs, outputs=x)


model_net_MobileNetv2 = unet_model(output_channels=OUTPUT_CLASSES)
# model_net_MobileNetv2.compile(optimizer=optimAdam)
