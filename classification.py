import os
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2


def parse_tfrecord(example_proto):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image[:16384], [128, 128, 1])
    image = tf.cast(image, tf.float32)
    image = (image + 1.0) / 2.0
    image = tf.tile(image, [1, 1, 3])

    label = tf.cast(parsed_features['label'], tf.int32)
    label = tf.one_hot(label, 5)
    return image, label


def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord)
    return dataset


def build_model(input_shape=(128, 128, 3), num_classes=5):
    # Define the input tensor
    input_tensor = Input(shape=input_shape)

    # Create the base InceptionResNetV2 model
    base_model = InceptionResNetV2(include_top=False,
                                   weights='imagenet',
                                   input_tensor=input_tensor)

    # Add global average pooling layer
    x = GlobalAveragePooling2D()(base_model.output)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer for the classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # Construct the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def main(tfrecord_path, dataset_name):
    dataset = load_dataset(tfrecord_path)
    dataset_size = sum(1 for _ in dataset)
    train_size = int(0.90 * dataset_size)

    train_dataset = dataset.take(train_size).batch(64).repeat()
    test_dataset = dataset.skip(train_size).batch(64).repeat()

    model = build_model()

    log_dir = os.path.join('logs',
                           dataset_name,
                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall(), AUC()])

    model.fit(train_dataset,
              epochs=20,
              steps_per_epoch=dataset_size // 64,
              callbacks=[tensorboard_callback],
              validation_data=test_dataset,
              validation_steps=dataset_size*0.10 // 64)

    model.save(f'models/classification/{dataset_name}.keras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    main(args.tfrecord_path, args.dataset_name)
