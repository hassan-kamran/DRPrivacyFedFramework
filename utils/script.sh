#!/bin/bash

# Loop through the postfix values
for postfix in {4..10}
do
    echo "Running training with postfix $postfix"

    # Place your command for creating a new Generator here
    # e.g., cp path/to/default_generator.keras models/generator_$postfix.keras

    # Place your command for creating a new Discriminator here
    # e.g., cp path/to/default_discriminator.keras models/discriminator_$postfix.keras

    # Run the training command
    python training.py --tfrecord_file_path data/clusters/diabetic-retinopath-128-16-labeled_part_$postfix.tfrecord \
                       --pretrained_generator_path models/pretraining/generator/generator_epoch_0002_pt.keras \
                       --pretrained_discriminator_path models/pretraining/discriminator/discriminator_epoch_0002_pt.keras \
                       --postfix $postfix \
                       --epochs 50
done
