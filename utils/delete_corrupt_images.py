import os
import argparse
import shutil
import tensorflow as tf
from tqdm import tqdm


def check_and_delete_corrupt_images(image_folder_path, backup_folder_path):
    # Initialize tqdm with the total number of files in the folder
    total_files = len(os.listdir(image_folder_path))
    with tqdm(total=total_files) as pbar:
        for filename in os.listdir(image_folder_path):
            filepath = os.path.join(image_folder_path, filename)
            try:
                image_data = tf.io.read_file(filepath)
                tf.io.decode_jpeg(image_data)
            except tf.errors.InvalidArgumentError as e:
                print(
                    f"Corrupt image detected & moved to backup:{filename} {e}")
                shutil.move(filepath, os.path.join(
                    backup_folder_path, filename))
            pbar.update(1)  # Update the progress bar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Delete corrupt images from a folder.')
    parser.add_argument('image_folder', type=str,
                        help='Path to the image folder')
    parser.add_argument('backup_folder', type=str,
                        help='Path to the backup folder')

    args = parser.parse_args()
    image_folder = args.image_folder
    backup_folder = args.backup_folder

    check_and_delete_corrupt_images(image_folder, backup_folder)
