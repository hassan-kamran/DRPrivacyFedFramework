# Standard library imports
from argparse import ArgumentParser
from os import path, walk
from typing import Tuple, Union

# Third-party imports
import numpy as np
from pandas import read_csv
import tensorflow as tf
from tensorflow import (
    Tensor, cond, convert_to_tensor,
    round, squeeze, uint8, ensure_shape
)
from tensorflow.data import Dataset
from tensorflow.image import (
    adjust_gamma, decode_jpeg, decode_png,
    resize, rgb_to_grayscale
)
from tensorflow.io import (
    TFRecordOptions, TFRecordWriter,
    read_file, serialize_tensor
)
from tensorflow.strings import regex_full_match
from tensorflow.train import Example, Features, Feature, Int64List, BytesList
from tf_clahe import clahe
from tqdm import tqdm


@tf.function
def image_processing(image_array: Union[np.ndarray, Tensor],
                     resolution: Tuple[int, int],
                     bin_size: int) -> Tensor:
    # Convert the input image array to a TensorFlow tensor with type uint8
    image = convert_to_tensor(image_array, dtype=uint8)
    # Resize the image to the target resolution
    resized_image = resize(image, resolution)
    # Convert the resized image to grayscale
    gray_image = rgb_to_grayscale(resized_image)
    # Remove the singleton dimension from the grayscale image
    gray_image = squeeze(gray_image)
    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe_image = clahe(gray_image, clip_limit=2.0, tile_grid_size=(8, 8))
    # Normalize the pixel values to range [0, 1]
    normalized_image = clahe_image / 255.0
    # Apply gamma correction to the image
    gamma_corrected_image = adjust_gamma(normalized_image, gamma=1.4)
    # Binning Image Pixels for Model Efficency
    binned_image = round(gamma_corrected_image * bin_size) / bin_size
    # Rescale the pixel values to range [-1, 1]
    scaled_image = binned_image * 2.0 - 1.0
    # Return the preprocessed image
    return scaled_image


def load_process(input: str, output: str, res: Tuple[int, int],
                 bin: int, label: str = None) -> None:
    # Read labels from CSV into a dictionary, if provided
    diabetes_dict = None
    if label:
        df = read_csv(label)
        diabetes_dict = dict(zip(df['image_name'], df['diabetes_level']))
    # List all image file paths
    image_paths = []
    for root, _, files in walk(input):
        for file_name in files:
            if file_name.endswith(('.png', '.jpg', '.jpeg', 'JPG')):
                image_paths.append(path.join(root, file_name))

    # Create a tf.data.Dataset from image paths
    image_paths_ds = Dataset.from_tensor_slices(image_paths)

    # Apply your image processing function
    processed_images_ds = image_paths_ds.map(
        lambda path: image_processing(
            ensure_shape(
                cond(
                    regex_full_match(path, r".*\.jpg"),
                    lambda: decode_jpeg(read_file(path), channels=3),
                    lambda: decode_png(read_file(path), channels=3)
                ),
                (None, None, 3)
            ),
            res,
            bin
        )
    )
    # Set compression options for TFRecord
    options = TFRecordOptions(compression_type='GZIP')
    # Write processed images to TFRecord
    with TFRecordWriter(output, options=options) as writer:
        for image_path, image_tensor in zip(image_paths,
                                            tqdm(processed_images_ds,
                                                 desc='ProcessingImages')):
            # Serialize the image tensor to byte string
            serialized_image = serialize_tensor(image_tensor).numpy()
            # Fetch the diabetes level if the dictionary is available
            if diabetes_dict:
                # Extract the image name from the image path
                # image_name = path.basename(image_path)
                image_name = path.splitext(path.basename(image_path))[0]
                # Fetch the diabetes level from the dictionary
                diabetes_level = diabetes_dict.get(
                    image_name, 0)  # Default to 0 if not found
                # Create a feature dictionary for the TFRecord
                feature = {
                    'image': Feature(
                        bytes_list=BytesList(value=[serialized_image])),
                    'label': Feature(
                        int64_list=Int64List(value=[diabetes_level]))
                }
                # Create an Example proto and serialize it to byte string
                example_proto = Example(
                    features=Features(feature=feature))
                writer.write(example_proto.SerializeToString())
            else:
                # If labels are not available, write only the serialized image
                writer.write(serialized_image)


if __name__ == "__main__":
    # Initialize the ArgumentParser object
    parser = ArgumentParser(
        description="Load and preprocess images, and save them to a TFRecord.")
    # Add command-line arguments
    parser.add_argument('input_folder', type=str,
                        help="Path to the folder containing images.")
    parser.add_argument('output_folder', type=str,
                        help="Path to the output TFRecord file.")
    parser.add_argument('resolution', type=int,
                        help="Resolution for both w and h of the image.")
    parser.add_argument('bin_size', type=int, help="Quantization level.")

    # Making labels optional
    parser.add_argument('labels', type=str, nargs='?', default=None,
                        help="Optional path to the CSV containing labels.")
    # Parse the command-line arguments
    args = parser.parse_args()
    # Call the load_process function with parsed arguments
    load_process(
        input=args.input_folder,
        output=args.output_folder,
        res=(args.resolution, args.resolution),
        bin=args.bin_size,
        label=args.labels
    )
