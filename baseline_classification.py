import os
import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.backend import epsilon
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.metrics import Precision, Recall, AUC


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


def main(tfrecord_path, dataset_name):
    dataset = load_dataset(tfrecord_path)
    dataset_size = sum(1 for _ in dataset)
    train_size = int(0.90 * dataset_size)
    train_dataset = dataset.take(train_size).batch(64).repeat()
    test_dataset = dataset.skip(train_size).batch(64)
    model = build_model()
    log_dir = os.path.join('logs',
                           dataset_name,
                           datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           MulticlassMetrics(5),
                           AUC(multi_label=True),
                           AUC(curve='PR', multi_label=True, name='auc_pr')])
    model.fit(train_dataset,
              epochs=5,
              steps_per_epoch=dataset_size // 64,
              callbacks=[tensorboard_callback])
    model.save(f'models/classification/{dataset_name}.keras')
    results = model.evaluate(test_dataset, steps=dataset_size*0.10 // 64)
    with open(f'logs/{dataset_name}/{dataset_name}.txt', 'w') as f:
        f.write(f'Loss: {results[0]}\n')
        f.write(f'Accuracy: {results[1]}\n')
        f.write(f'Precision: {results[2]}\n')
        f.write(f'Recall: {results[3]}\n')
        f.write(f'F1 Score: {results[4]}\n')
        f.write(f'AUC: {results[5]}\n')
        f.write(f'PR AUC: {results[6]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    main(args.tfrecord_path, args.dataset_name)
