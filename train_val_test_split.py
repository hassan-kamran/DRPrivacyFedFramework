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

def split_tfrecord(tfrecord_path, train_ratio, test_ratio, val_ratio):
    # Read and store original data
    dataset = TFRecordDataset(tfrecord_path, compression_type="GZIP")
    dataset = dataset.map(_parse_image_function)

    data = [(image.numpy(), label.numpy()) for image, label in dataset]
    np.random.shuffle(data)

    # Calculate the size of each split
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)

    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    # Create and write split TFRecords
    base_name = splitext(tfrecord_path)[0]
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    
    for dataset_name, dataset_data in zip(['train', 'test', 'val'], [train_data, test_data, val_data]):
        split_tfrecord_path = f"{base_name}_{dataset_name}.tfrecord"
        with TFRecordWriter(split_tfrecord_path, options=options) as writer:
            for image, label in dataset_data:
                _write_example(writer, image, label)
        print(f"Created: {split_tfrecord_path}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('tfrecord_path', type=str)
    parser.add_argument('train_ratio', type=float)
    parser.add_argument('test_ratio', type=float)
    parser.add_argument('val_ratio', type=float)
    args = parser.parse_args()

    # Ensure that the ratios sum up to 1
    total_ratio = args.train_ratio + args.test_ratio + args.val_ratio
    if total_ratio != 1.0:
        raise ValueError("The sum of train, test, and val ratios must be 1")

    split_tfrecord(args.tfrecord_path, args.train_ratio, args.test_ratio, args.val_ratio)
