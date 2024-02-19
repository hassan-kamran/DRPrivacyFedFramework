from tensorflow.keras.layers import (
    Layer, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, LeakyReLU
)


class TransposedConvLayer(Layer):
    def __init__(self, filters, kernel_size,
                 strides, padding, momentum, **kwargs):
        super(TransposedConvLayer, self).__init__(**kwargs)
        self.transposed_conv = Conv2DTranspose(filters,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               use_bias=False)
        self.batch_norm = BatchNormalization(momentum=momentum)
        self.relu = ReLU()

    def call(self, input_tensor):
        x = self.transposed_conv(input_tensor)
        x = self.batch_norm(x)
        return self.relu(x)


class ConvLayer(Layer):
    def __init__(self, filters, kernel_size,
                 strides, padding, momentum, alpha, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.conv = Conv2D(filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           use_bias=False)
        self.batch_norm = BatchNormalization(momentum=momentum)
        self.leaky_relu = LeakyReLU(alpha=alpha)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.batch_norm(x)
        return self.leaky_relu(x)
