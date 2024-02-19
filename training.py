# Standard library imports
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import join
from re import match, findall

# Third-party library imports
import matplotlib.pyplot as plt
from tensorflow import (
    expand_dims, ones_like,
    zeros_like, shape,
    GradientTape, float32,
    function, string, int64)
from tensorflow.data import TFRecordDataset
from tensorflow.io import parse_tensor, FixedLenFeature, parse_single_example
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.random import normal
from tensorflow.keras.optimizers.legacy import Adam
from numpy import array, int32

# Local application/library specific imports
from model_cgan import build_generator, build_discriminator
from custom_layers import TransposedConvLayer, ConvLayer


BATCH_SIZE = 64
NOISE_DIM = 200
LEARNING_RATE = 0.0002
BETA_1 = 0.5
EPOCHS = 1  # Set this to 10 for pretraining and 1000 for training
SAVE_MODEL_INTERVAL = 1


# Function to parse the TFRecord file and get the image
def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using a dictionary describing the data
    feature_description = {
        'image': FixedLenFeature([], string),
        'label': FixedLenFeature([], int64),
    }
    example = parse_single_example(example_proto, feature_description)
    image_tensor = parse_tensor(example['image'], out_type=float32)
    image_tensor = expand_dims(image_tensor, axis=-1)
    label = example['label']
    return image_tensor, label


# Create dataset from TFRecord
def create_dataset_from_tfrecord(tfrecord_path, batch_size):
    raw_image_dataset = TFRecordDataset(
        tfrecord_path, compression_type="GZIP")
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    return parsed_image_dataset.batch(batch_size)


# Define loss functions
def discriminator_loss(real_output, fake_output):
    # Smoothing factor of 0.9 for label smoothing
    real_loss = binary_crossentropy(ones_like(real_output) * 0.9,
                                    real_output)
    fake_loss = binary_crossentropy(zeros_like(fake_output),
                                    fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return binary_crossentropy(ones_like(fake_output),
                               fake_output)


@function
def train_step(images, labels, generator, discriminator,
               gen_optimizer, disc_optimizer, noise_dim):
    noise = normal([shape(images)[0], noise_dim])

    with GradientTape() as gen_tape, GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator.
                                                    trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_of_generator,
                                      generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                       discriminator.trainable_variables))

    return gen_loss, disc_loss


def save_images(generator, test_input, test_labels, epoch, start_epoch, postfix):
    predictions = generator([test_input, test_labels], training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'imgs/image_at_epoch_{epoch + start_epoch + 1:04d}_t_{postfix}.png')
    plt.close()


def find_latest_model(model_type, models_dir='models/'):
    latest_epoch = 0
    latest_model_file = None
    pattern = rf'{model_type}_epoch_([0-9]{{4}})_t.keras'

    for file_name in listdir(models_dir):
        if match(pattern, file_name):
            epoch_num = int(findall(pattern, file_name)[0])
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_model_file = join(models_dir, file_name)

    return latest_model_file, latest_epoch


def transfer_weights(pretrained_model, new_model):
    for layer in new_model.layers:
        if layer.name in pretrained_model.layers:
            pretrained_layer = pretrained_model.get_layer(layer.name)
            if layer.get_config() == pretrained_layer.get_config():
                layer.set_weights(pretrained_layer.get_weights())


# Define the training function
def train(tfrecord_file_path, epochs, noise_dim, batch_size,
          learning_rate, beta_1, save_model_interval,
          pretrained_generator_path, pretrained_discriminator_path, postfix):

    makedirs('models', exist_ok=True)
    makedirs('imgs', exist_ok=True)

    generator_model_file, gen_epoch = find_latest_model('generator')
    discriminator_model_file, disc_epoch = find_latest_model('discriminator')

    if generator_model_file and discriminator_model_file:
        print(f"Loading Epoch Gen:{gen_epoch}, Dis:{disc_epoch}")
        try:
            custom_objects = {'TransposedConvLayer': TransposedConvLayer,
                              'ConvLayer': ConvLayer}
            generator = load_model(generator_model_file,
                                   custom_objects=custom_objects)
            discriminator = load_model(discriminator_model_file,
                                       custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    else:
        print("Create New Generator and Discriminaotr")
        # Load pretrained models (original architecture)

        custom_objects = {'TransposedConvLayer': TransposedConvLayer,
                          'ConvLayer': ConvLayer}
        pretrained_generator = load_model(pretrained_generator_path,
                                          custom_objects=custom_objects)
        pretrained_discriminator = load_model(pretrained_discriminator_path,
                                              custom_objects=custom_objects)
        generator = build_generator(noise_dim)
        discriminator = build_discriminator()

        # Transfer weights from pretrained models to new conditional models
        transfer_weights(pretrained_generator, generator)
        transfer_weights(pretrained_discriminator, discriminator)

    # Define the optimizers
    generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
    discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)

    # Prepare dataset
    dataset = create_dataset_from_tfrecord(tfrecord_file_path, batch_size)

    # Seed noise for image saving and labels
    seed_noise = normal([16, noise_dim])
    test_labels = array([i % 6 for i in range(16)],
                        dtype=int32).reshape(-1, 1)

    # Determine the starting epoch based on loaded models
    start_epoch = max(gen_epoch, disc_epoch)

    for epoch in range(epochs):
        adjusted_epoch = epoch + start_epoch
        for image_batch, label_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, label_batch,
                                             generator, discriminator,
                                             generator_optimizer,
                                             discriminator_optimizer,
                                             noise_dim)

        generator.save(
            f'models/generator_epoch_{adjusted_epoch + 1:04d}_t_{postfix}.keras')
        discriminator.save(
            f'models/discriminator_epoch_{adjusted_epoch + 1:04d}_t_{postfix}.keras')
        # Inside your training loop
        save_images(generator, seed_noise, test_labels, epoch, start_epoch, postfix)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Train a DCGAN model on a given TFRecord file')
    parser.add_argument('--tfrecord_file_path', type=str, required=True,
                        help='Path to the TFRecord file')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--noise_dim', type=int, default=NOISE_DIM,
                        help='Dimension of the noise vector')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--beta_1', type=float, default=BETA_1,
                        help='Beta1 value for Adam optimizer')
    parser.add_argument('--save_model_interval', type=int,
                        default=SAVE_MODEL_INTERVAL,
                        help='Interval (in epochs) to save the model')
    parser.add_argument('--pretrained_generator_path', type=str, default=None,
                        help='Path to the pretrained generator model')
    parser.add_argument('--pretrained_discriminator_path', type=str,
                        default=None,
                        help='Path to the pretrained discriminator model')
    parser.add_argument('--postfix', type=str, help='cluster number')
    args = parser.parse_args()

    train(args.tfrecord_file_path, args.epochs, args.noise_dim,
          args.batch_size, args.learning_rate, args.beta_1,
          args.save_model_interval,
          args.pretrained_generator_path,
          args.pretrained_discriminator_path,
          args.postfix)
