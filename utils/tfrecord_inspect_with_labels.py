from argparse import ArgumentParser
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to parse the tf.Example format


def _parse_image_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(
        example_proto, feature_description)

    image_tensor = tf.io.parse_tensor(
        parsed_features['image'], out_type=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=-1)

    return image_tensor, parsed_features['label']


def create_dataset_from_tfrecord(tfrecord_path, num_images=8):
    raw_image_dataset = tf.data.TFRecordDataset(
        tfrecord_path, compression_type="GZIP")
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    return parsed_image_dataset.take(num_images)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tfrecord_path', type=str)
    args = parser.parse_args()
    dataset = create_dataset_from_tfrecord(args.tfrecord_path)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # 2x4 grid
    for i, (img_tensor, label) in enumerate(dataset):
        row = i // 4
        col = i % 4
        img = img_tensor.numpy().squeeze()
        min_val = img.min()
        max_val = img.max()
        img_rescaled = (img + 1) / 2.0
        axes[row, col].imshow(img_rescaled, cmap='gray')
        axes[row, col].set_title(
            f"Min: {min_val}, Max: {max_val}, Label: {label.numpy()}")
    plt.show()
