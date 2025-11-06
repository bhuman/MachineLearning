from __future__ import annotations

import keras
import parameters as params



def model_kick_in_without_softmax() -> keras.Model:
    input_image = keras.Input(shape=(256, 256, 3))
    input_distance = keras.layers.Input(shape=(1,), name="distance_input")

    x = keras.layers.Conv2D(8, (5, 5), (2, 2), "same", use_bias=False)(input_image)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(16, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(16, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(16, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    distance_dense = keras.layers.Dense(8)(input_distance)
    distance_dense = keras.layers.ReLU()(distance_dense)

    merged_layer = keras.layers.concatenate([x, distance_dense])

    x = keras.layers.Dense(24)(merged_layer)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(3, name="Output_Layer")(x)

    return keras.Model(inputs=[input_image, input_distance], outputs=x)


def model_standby_to_ready_without_softmax() -> keras.Model:
    input_image = keras.Input(shape=(256, 200, 3))
    input_distance = keras.layers.Input(shape=(1,), name="distance_input")

    x = keras.layers.Conv2D(8, (5, 5), (2, 2), "same", use_bias=False)(input_image)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(16, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(16, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (2, 2), "same", use_bias=True)(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.SeparableConv2D(32, (3, 3), (1, 1), "same", use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Flatten()(x)
    distance_dense = keras.layers.Dense(8)(input_distance)
    distance_dense = keras.layers.ReLU()(distance_dense)

    merged_layer = keras.layers.concatenate([x, distance_dense])

    x = keras.layers.Dense(24)(merged_layer)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(2, name="Output_Layer")(x)

    return keras.Model(inputs=[input_image, input_distance], outputs=x)
