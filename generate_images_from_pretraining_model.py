import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from model_dcgan import TransposedConvLayer, ConvLayer
from argparse import ArgumentParser


def save_images(generator, test_input, output_path):
    predictions = generator(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--generator_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    custom_objects = {'TransposedConvLayer': TransposedConvLayer,
                      'ConvLayer': ConvLayer}
    generator = load_model(args.generator_path,
                           custom_objects=custom_objects)

    latent_vector = tf.random.normal([16, 200])

    save_images(generator, latent_vector, args.output_path)
