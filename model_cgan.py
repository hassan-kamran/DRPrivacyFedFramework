import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Concatenate,
    Dense, Reshape, Conv2DTranspose, Activation
)
from custom_layers import TransposedConvLayer, ConvLayer


# Modified build_generator to accept labels
def build_generator(latent_dim=200):
    # Additional input for the label
    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(input_dim=2,
                                output_dim=latent_dim)(label_input)
    label_embedding = Flatten()(label_embedding)
    # Original generator's latent input
    latent_input = Input(shape=(latent_dim,))
    x = Concatenate()([latent_input, label_embedding])
    x = Dense(4*4*1024, use_bias=False)(x)
    x = Reshape((4, 4, 1024))(x)
    # Transposed convolutional layers
    x = TransposedConvLayer(512, kernel_size=4, strides=2,
                            padding='same', momentum=0.1)(x)
    x = TransposedConvLayer(256, kernel_size=4, strides=2,
                            padding='same', momentum=0.1)(x)
    x = TransposedConvLayer(128, kernel_size=4, strides=2,
                            padding='same', momentum=0.1)(x)
    x = TransposedConvLayer(64, kernel_size=4, strides=2,
                            padding='same', momentum=0.1)(x)
    x = TransposedConvLayer(32, kernel_size=4, strides=2,
                            padding='same', momentum=0.1)(x)
    # Output layer
    x = Conv2DTranspose(1, kernel_size=4, padding='same', use_bias=False)(x)
    outputs = Activation('tanh')(x)
    model = Model([latent_input, label_input], outputs, name='generator')
    return model


# Modified build_discriminator to accept labels with images
def build_discriminator(image_shape=(128, 128, 1)):
    # Image input
    image_input = Input(shape=image_shape)
    # Additional input for the label
    label_input = Input(shape=(1,), dtype='int32')
    label_embedding = Embedding(input_dim=2,
                                output_dim=np.prod(image_shape))(label_input)
    label_embedding = Flatten()(label_embedding)
    label_embedding = Reshape(image_shape)(label_embedding)
    # Combine image and label
    combined_input = Concatenate(axis=-1)([image_input, label_embedding])
    # Convolutional layers
    x = ConvLayer(64, kernel_size=4, strides=2,
                  padding='same', momentum=0.1, alpha=0.2)(combined_input)
    x = ConvLayer(128, kernel_size=4, strides=2,
                  padding='same', momentum=0.1, alpha=0.2)(x)
    x = ConvLayer(256, kernel_size=4, strides=2,
                  padding='same', momentum=0.1, alpha=0.2)(x)
    x = ConvLayer(512, kernel_size=4, strides=2,
                  padding='same', momentum=0.1, alpha=0.2)(x)
    x = ConvLayer(1024, kernel_size=4, strides=2,
                  padding='same', momentum=0.1, alpha=0.2)(x)
    # Flatten and output layer
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model([image_input, label_input], outputs, name='discriminator')
    return model
