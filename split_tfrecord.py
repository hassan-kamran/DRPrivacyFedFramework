
from argparse import ArgumentParser
from os.path import splitext
import tensorflow as tf
import numpy as np
from tensorflow.data import TFRecordDataset
from tensorflow.io import FixedLenFeature, parse_single_example, TFRecordWriter, parse_tensor
from tensorflow.train import BytesList, Int64List, Feature, Features, Example

# Function to parse the tf.Example format
def _parse_image_function(example_proto):
    feature_description = {
        'image': FixedLenFeature([], tf.string),
        'label': FixedLenFeature([], tf.int64),
    }
    parsed_features = parse_single_example(example_proto, feature_description)
    image_tensor = parse_tensor(parsed_features['image'], out_type=tf.float32)
    return image_tensor, parsed_features['label']

# Function to write an example to a TFRecord
def _write_example(writer, image, label):
    serialized_image = tf.io.serialize_tensor(image).numpy()
    feature = {
        'image': Feature(bytes_list=BytesList(value=[serialized_image])),
        'label': Feature(int64_list=Int64List(value=[label]))
    }
    example = Example(features=Features(feature=feature))
    writer.write(example.SerializeToString())

def split_tfrecord(tfrecord_path, num_splits):
    # Read and store original data
    dataset = TFRecordDataset(tfrecord_path, compression_type="GZIP")
    dataset = dataset.map(_parse_image_function)

    data = [(image.numpy(), label.numpy()) for image, label in dataset]
    np.random.shuffle(data)

    # Calculate the size of each split
    split_size = len(data) // num_splits
    split_data = [data[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    # Handle any remaining data by adding it to the last split
    if len(data) % num_splits != 0:
        split_data[-1].extend(data[num_splits * split_size:])

    # Create and write split TFRecords
    base_name = splitext(tfrecord_path)[0]
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    for i, split in enumerate(split_data):
        split_tfrecord_path = f"{base_name}_part_{i + 1}.tfrecord"
        with TFRecordWriter(split_tfrecord_path, options=options) as writer:
            for image, label in split:
                _write_example(writer, image, label)
        print(f"Created: {split_tfrecord_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tfrecord_path', type=str)
    parser.add_argument('num_splits', type=int)
    args = parser.parse_args()

    split_tfrecord(args.tfrecord_path, args.num_splits)
