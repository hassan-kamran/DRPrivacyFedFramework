import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from model_dcgan import TransposedConvLayer, ConvLayer
from argparse import ArgumentParser

def save_images(generator, test_input, test_labels, output_path, epoch=0, start_epoch=0, postfix=''):
    predictions = generator([test_input, test_labels], training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'{output_path}/image_at_epoch_{epoch + start_epoch + 1:04d}_t_{postfix}.png')
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--generator_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    custom_objects = {'TransposedConvLayer': TransposedConvLayer,
                      'ConvLayer': ConvLayer}
    generator = load_model(args.generator_path, custom_objects=custom_objects)

    noise_dim = 200  # Adjust based on your model's requirement
    latent_vector = tf.random.normal([16, noise_dim])

    # Scalar diabetes levels
    diabetes_levels = np.array([i % 5 for i in range(16)])  # 0, 1, 2, 3, 4 levels

    save_images(generator, latent_vector, diabetes_levels, args.output_path)
