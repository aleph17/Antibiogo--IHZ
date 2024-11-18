from tensorflow_examples.models.pix2pix import pix2pix
from ssl import _create_default_https_context, _create_unverified_context
from utils import IMG_SIZE, OUTPUT_CLASSES, initial_bias
import tensorflow as tf
from modelclass import CustomModel


def hungarian():
    # _create_default_https_context = _create_unverified_context
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    model = tf.keras.applications.MobileNetV2(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)
    x = model(inputs)
    # print('after mobilenet', x.shape)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), activation='relu')(x)
    # print('after depthwise', x.shape)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    # print('after Global max pooling', x.shape)
    x = tf.keras.layers.Dense(48, bias_initializer=tf.keras.initializers.Constant(initial_bias))(x)
    # print('after dense 48', x.shape)
    x = tf.keras.layers.Reshape((16,3))(x)
    # print('after reshape', outputs.shape)
    return CustomModel(inputs=inputs, outputs=x)


model = hungarian()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
# print(model.summary())
