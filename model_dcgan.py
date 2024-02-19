from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape,
    Conv2DTranspose, Activation, Flatten)
from custom_layers import TransposedConvLayer, ConvLayer


def build_generator(latent_dim=200):
    inputs = Input(shape=(latent_dim,))
    x = Dense(4*4*1024, use_bias=False)(inputs)
    x = Reshape((4, 4, 1024))(x)
    x = TransposedConvLayer(512, kernel_size=4, strides=2, padding='same',
                            momentum=0.1)(x)
    x = TransposedConvLayer(256, kernel_size=4, strides=2, padding='same',
                            momentum=0.1)(x)
    x = TransposedConvLayer(128, kernel_size=4, strides=2, padding='same',
                            momentum=0.1)(x)
    x = TransposedConvLayer(64, kernel_size=4, strides=2, padding='same',
                            momentum=0.1)(x)
    x = TransposedConvLayer(32, kernel_size=4, strides=2, padding='same',
                            momentum=0.1)(x)
    x = Conv2DTranspose(1, kernel_size=4, padding='same', use_bias=False)(x)
    outputs = Activation('tanh')(x)
    model = Model(inputs, outputs, name='generator')
    return model


def build_discriminator(image_shape=(128, 128, 1)):
    inputs = Input(shape=image_shape)
    x = ConvLayer(64, kernel_size=4, strides=2, padding='same',
                  momentum=0.1, alpha=0.2)(inputs)
    x = ConvLayer(128, kernel_size=4, strides=2, padding='same', momentum=0.1,
                  alpha=0.2)(x)
    x = ConvLayer(256, kernel_size=4, strides=2, padding='same', momentum=0.1,
                  alpha=0.2)(x)
    x = ConvLayer(512, kernel_size=4, strides=2, padding='same', momentum=0.1,
                  alpha=0.2)(x)
    x = ConvLayer(1024, kernel_size=4, strides=2, padding='same', momentum=0.1,
                  alpha=0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name='discriminator')
    return model
