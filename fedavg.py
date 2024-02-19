import os
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.backend import epsilon
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Precision, Recall, AUC
import numpy as np
import glob


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


def build_model():
    base_model = tf.keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(128, 128, 3))

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.05)(x)

    predictions = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    return model


class MulticlassMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='multiclass_metrics', **kwargs):
        super(MulticlassMetrics, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision = Precision(class_id=None, name='precision')
        self.recall = Recall(class_id=None, name='recall')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1_score = 2 * (precision * recall) / (precision + recall + epsilon())

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()


class CustomFederatedLearning:
    def __init__(self, dataset, model, num_clients, personalized=False):
        self.dataset = dataset
        self.model = model
        self.num_clients = num_clients
        self.personalized = personalized
        self.client_datasets = self.split_dataset_for_clients()

    def split_dataset_for_clients(self):
        client_datasets = []
        dataset_size = sum(1 for _ in self.dataset)
        per_client_size = dataset_size // self.num_clients
        for i in range(self.num_clients):
            client_dataset = self.dataset.skip(per_client_size * i).take(per_client_size)
            client_datasets.append(client_dataset)
        return client_datasets

    def train(self, epochs=5, batch_size=64):
        for client_id, client_dataset in enumerate(self.client_datasets):
            print(f"Training on Client {client_id}")
            train_size = int(0.90 * sum(1 for _ in client_dataset))
            train_dataset = client_dataset.take(train_size).batch(batch_size).repeat()
            test_dataset = client_dataset.skip(train_size).batch(batch_size)

            # Setting up TensorBoard for logging
            log_dir = f"logs/client_{client_id}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

            if self.personalized:
                client_model = build_model()
            else:
                client_model = self.model

            client_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy',
                                          MulticlassMetrics(5)],
                                 callbacks=[tensorboard_callback])

            client_model.fit(train_dataset, epochs=epochs,
                             steps_per_epoch=train_size // batch_size)
            results = client_model.evaluate(test_dataset,
                                            steps=(sum(1 for _ in client_dataset) - train_size) // batch_size)
            print(f"Results for Client {client_id}: {results}")

            if self.personalized:
                client_model.save(f'models/classification/client_{client_id}.keras')


def main(tfrecord_dir, num_clients, personalized):
    tfrecord_files = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    if not tfrecord_files:
        raise ValueError("No TFRecord files found in the specified directory.")

    for tfrecord_file in tfrecord_files:
        dataset_name = os.path.basename(tfrecord_file).split('.')[0]
        dataset = load_dataset(tfrecord_file)
        model = build_model()  # Global model

        fl_trainer = CustomFederatedLearning(dataset,
                                             model,
                                             num_clients,
                                             personalized)
        print(f"Training on dataset: {dataset_name}")
        fl_trainer.train()
    # Save the global model
    model.save('models/global_model.keras')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_dir', type=str, required=True)
    parser.add_argument('--num_clients', type=int, required=True)
    parser.add_argument('--personalized', action='store_true')
    args = parser.parse_args()

    main(args.tfrecord_dir, args.num_clients, args.personalized)
