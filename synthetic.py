import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
from model_dcgan import TransposedConvLayer, ConvLayer
from argparse import ArgumentParser


def reset_tensorflow_state():
    tf.keras.backend.clear_session()
    tf.random.set_seed(None)


def save_image(generator, noise_dim, test_label, output_path, seed):
    # Reset TensorFlow state
    reset_tensorflow_state()

    # Set a new seed for each image
    tf.random.set_seed(seed)

    random_latent_vector = tf.random.normal([1, noise_dim])
    prediction = generator([random_latent_vector, np.array([test_label])],
                           training=False)[0]

    # Convert the prediction to a PIL Image and save
    image = Image.fromarray(np.uint8((prediction[:, :, 0] * 0.5 + 0.5) * 255),
                            'L')
    image.save(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--generator_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    args = parser.parse_args()

    custom_objects = {'TransposedConvLayer': TransposedConvLayer,
                      'ConvLayer': ConvLayer}
    generator = load_model(args.generator_path, custom_objects=custom_objects)

    noise_dim = 200  # Adjust based on your model's requirement
    num_images_per_category = 1000

    for category in range(5):  # Diabetes categories 0, 1, 2, 3, 4
        category_folder = os.path.join(args.output_folder,
                                       f'{category}')
        os.makedirs(category_folder, exist_ok=True)

        for i in range(num_images_per_category):
            image_path = os.path.join(category_folder, f'image_{i:04d}.png')
            save_image(generator, noise_dim, category, image_path, i)

    print("Image generation complete.")
