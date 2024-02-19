# Differentially Private Retina Generative Adversarial Network

## Description

This project aims to build a Generative Adversarial Network (GAN) focused on generating synthetic medical images, specifically of the retina. It also includes functionalities for data preprocessing, semantic segmentation, and performance evaluation.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [File Structure](#file-structure)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

1. Clone the repository.
    ```bash
    git clone https://gitlab.com/your-username/your-repo.git
    ```
2. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- **Data Preprocessing**
    ```bash
    python preprocessing.py --arg1 --arg2
    ```

- **Training the GAN**
    ```bash
    python retina_gan.py --arg1 --arg2
    ```

- **Semantic Segmentation**
    ```bash
    python semantic_segmentation.py --arg1 --arg2
    ```

- **Inspect TFRecords**
    ```bash
    python tfrecord_inspect_no_labels.py --arg1
    python tfrecord_inspect_with_labels.py --arg2
    ```

- **Delete Corrupt Images**
    ```bash
    python delete_corrupt_images.py --arg1 --arg2
    ```

## File Structure

- `custom_layers.py`: Contains custom TensorFlow layers.
- `delete_corrupt_images.py`: Deletes corrupt image files from a folder.
- `preprocessing.py`: Handles data preprocessing.
- `retina_gan.py`: Main script for the GAN model.
- `semantic_segmentation.py`: Contains code for semantic segmentation tasks.
- `tfrecord_inspect_no_labels.py`: Inspects TFRecords without labels.
- `tfrecord_inspect_with_labels.py`: Inspects TFRecords with labels.
- `training.py`: Script for training models.

## Contributing

Feel free to fork the project and submit a pull request.

## License

MIT License. See the [LICENSE](LICENSE) file for details.
