# Standard library imports
from argparse import ArgumentParser
from os import listdir, remove, makedirs
from os.path import join

# Third-party imports
from numpy import array
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# TensorFlow imports
from tensorflow import convert_to_tensor, squeeze, uint8, round, function, cast
from tensorflow.image import (adjust_gamma,
                              resize,
                              rgb_to_grayscale,
                              decode_image)
from tensorflow.io import (read_file,
                           serialize_tensor,
                           TFRecordOptions,
                           TFRecordWriter)
from tensorflow.train import Example, Features, Feature, BytesList, Int64List
from tensorflow.errors import InvalidArgumentError
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Custom or additional third-party imports
from tf_clahe import clahe

# Constants
IMAGE_SIZE = (128, 128)
NUM_CLUSTERS = 5
CLUSTER_FOLDER = 'clusters'


# Function to delete corrupt images
def delete_corrupt_images(folder_path):
    for filename in tqdm(listdir(folder_path),
                         desc='Checking images'):
        filepath = join(folder_path, filename)
        try:
            _ = decode_image(read_file(filepath))
        except InvalidArgumentError:
            remove(filepath)
            print(f"Deleted corrupt image: {filepath}")


# Image processing function
@function
def image_processing(image_array, resolution=IMAGE_SIZE, bin_size=256):
    if image_array.dtype == 'float32':
        image_array = cast(image_array, dtype=uint8)
    image = convert_to_tensor(image_array, dtype=uint8)
    resized_image = resize(image, resolution)
    gray_image = rgb_to_grayscale(resized_image)
    gray_image = squeeze(gray_image)
    clahe_image = clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8))
    normalized_image = clahe_image / 255.0
    gamma_corrected_image = adjust_gamma(normalized_image, gamma=1.4)
    binned_image = round(gamma_corrected_image * bin_size) / bin_size
    scaled_image = binned_image * 2.0 - 1.0
    return scaled_image


# Semantic segmentation and preprocessing
def process_images(image_folder, num_clusters, diabetes_dict):
    images = []
    filenames = []
    for filename in tqdm(listdir(image_folder), desc='Loading images'):
        filepath = join(image_folder, filename)
        image = load_img(filepath, target_size=IMAGE_SIZE)
        images.append(img_to_array(image))
        filenames.append(filename)
    images = array(images, dtype="float32")

    # Feature extraction
    model = EfficientNetB3(include_top=False,
                           weights='imagenet',
                           pooling='avg')
    features = model.predict(images)

    # PCA for dimensionality reduction
    pca = PCA(n_components=256)
    features_reduced = pca.fit_transform(features)

    # Standardizing the features
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features_reduced)

    # GMM clustering
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    clusters = gmm.fit_predict(features_standardized)

    # Preprocess images
    preprocessed_images = [image_processing(image).numpy() for image in images]

    # Map images to their diabetes levels
    diabetes_levels = [diabetes_dict.get(filename,
                                         0) for filename in filenames]

    return preprocessed_images, clusters, diabetes_levels


# Function to write TFRecord
def write_tfrecord(images, labels, output_filename, diabetes_dict):
    options = TFRecordOptions(compression_type="GZIP")
    with TFRecordWriter(output_filename, options=options) as writer:
        for image, label in zip(images, labels):
            # Retrieve diabetes level from dictionary using label
            diabetes_level = diabetes_dict.get(label, 0)

            # Create a feature dictionary for the TFRecord
            feature = {
                'image': Feature(bytes_list=BytesList(
                    value=[serialize_tensor(image).numpy()])),
                'label': Feature(int64_list=Int64List(
                    value=[diabetes_level]))
            }

            # Create an Example proto and serialize it to byte string
            example = Example(features=Features(feature=feature))
            writer.write(example.SerializeToString())


def main(image_folder,
         labels_path):
    # Step 1: Delete corrupt images
    delete_corrupt_images(image_folder)

    # Step 2: Load diabetes levels
    df = read_csv(labels_path)
    diabetes_dict = dict(zip(df['image_name'],
                             df['diabetes_level']))

    # Step 3: Semantic Segmentation and Preprocessing
    processed_images, labels, diabetes_levels = process_images(image_folder,
                                                               NUM_CLUSTERS,
                                                               diabetes_dict)

    # Create the clusters folder
    makedirs(CLUSTER_FOLDER, exist_ok=True)

    # Group images by label for TFRecord writing
    for label in set(labels):
        label_specific_images = []
        label_specific_diabetes_levels = []
        for image, image_label, diabetes_level in zip(processed_images,
                                                      labels,
                                                      diabetes_levels):
            if image_label == label:
                label_specific_images.append(image)
                label_specific_diabetes_levels.append(diabetes_level)

        label_specific_tfrecord = join(CLUSTER_FOLDER,
                                       f'cluster_{label}.tfrecord')
        write_tfrecord(label_specific_images,
                       label_specific_diabetes_levels,
                       label_specific_tfrecord,
                       diabetes_dict)

    print(f"Data saved in TFRecord format in the directory: {CLUSTER_FOLDER}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Process images and save them in TFRecord format.")

    parser.add_argument('--input_folder',
                        type=str,
                        help="Path to the folder containing images.")

    parser.add_argument('--labels',
                        type=str,
                        help="Optional path to the CSV containing labels.",
                        default=None)

    args = parser.parse_args()

    main(args.input_folder,
         args.labels)
