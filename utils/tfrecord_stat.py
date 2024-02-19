import argparse
from tensorflow.data import TFRecordDataset
from tensorflow.io import FixedLenFeature, parse_single_example
from tensorflow import int64, string
from collections import Counter


def _parse_dataset(example_proto):
    features_description = {
        'image': FixedLenFeature([], string),
        'label': FixedLenFeature([], int64),
    }
    parsed_features = parse_single_example(
        example_proto,
        features_description
    )
    return parsed_features['label']


def load_tfrecord_and_count(tfrecord_path):
    raw_dataset = TFRecordDataset(tfrecord_path, compression_type='GZIP')
    parsed_dataset = raw_dataset.map(_parse_dataset)
    label_list = [label.numpy()
                  for label in parsed_dataset]  # Convert to NumPy here

    # Counting total images and label distribution
    total_images = len(label_list)
    label_count = Counter(label_list)

    return total_images, label_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count images and label distribution in TFRecord.')
    parser.add_argument('tfrecord_path', type=str, help='TFRecord file path')

    args = parser.parse_args()
    tfrecord_path = args.tfrecord_path

    total_images, label_count = load_tfrecord_and_count(tfrecord_path)

    print(f"Total Images in TFRecord: {total_images}")
    print(f"Label Distribution: {label_count}")
