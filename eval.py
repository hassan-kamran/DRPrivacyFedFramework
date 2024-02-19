import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import argparse
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam


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

    label = tf.cast(parsed_features['label'], tf.int32)
    label = tf.one_hot(label, 5)
    return image, label


def load_dataset(tfrecord_path, batch_size=64):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='GZIP')
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.batch(batch_size)  # Add batching
    return dataset


def build_model(input_shape=(128, 128, 1), num_classes=5):
    inputs = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(filters=64, kernel_size=(7, 7),
               strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
        shortcut = x

        # First convolution
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Second convolution
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)

        # Adjusting the shortcut path with a 1x1 convolution
        shortcut = Conv2D(filters, (1, 1), strides=strides,
                          padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

        # Adding the shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    # Residual Blocks
    for _ in range(2):
        x = residual_block(x, 64)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(2):
        x = residual_block(x, 128)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    for _ in range(2):
        x = residual_block(x, 256)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
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
        f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

        return {'precision': precision, 'recall': recall, 'f1_score': f1_score}

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()


def evaluate_model(tfrecord_path, keras_model_path, num_classes):
    # Load dataset
    dataset = load_dataset(tfrecord_path)

    # Load model architecture and weights
    try:
        model = load_model(keras_model_path, custom_objects={'MulticlassMetrics': lambda: MulticlassMetrics(num_classes)}, compile=False)
    except ValueError:
        # Build model and load weights if loading fails
        model = build_model(input_shape=(128, 128, 1), num_classes=num_classes)
        model.load_weights(keras_model_path)

    # Recompile the model
    model.compile(optimizer=Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', MulticlassMetrics(num_classes)])

    # Evaluate the model
    results = model.evaluate(dataset, return_dict=True)

    # Additional metrics: AUC and AUC PR
    auc_metric = AUC()
    auc_pr_metric = AUC(curve='PR')

    for images, labels in dataset:
        predictions = model.predict(images)
        auc_metric.update_state(labels, predictions)
        auc_pr_metric.update_state(labels, predictions)

    results['auc'] = auc_metric.result().numpy()
    results['auc_pr'] = auc_pr_metric.result().numpy()

    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description='Evaluate a Keras model using a TFRecord dataset.')
    parser.add_argument('tfrecord_path', type=str, help='Path to the TFRecord file.')
    parser.add_argument('keras_model_path', type=str, help='Path to the Keras model file.')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes in the dataset.')
    args = parser.parse_args()

    results = evaluate_model(args.tfrecord_path, args.keras_model_path, args.num_classes)
    for key, value in results.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    main()
