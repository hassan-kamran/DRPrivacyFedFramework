#!/bin/bash

# Assuming you have 10 clusters. Replace 10 with your actual number of clusters.
for cluster_num in $(seq -w 1 10)
do
    # Remove leading zeros for t value
    t_value=$((10#$cluster_num))

    echo "Processing cluster $cluster_num with t value $t_value"
    python synthetic.py --generator_path "models/clusters/$cluster_num/generator/generator_epoch_0050_t_${t_value}.keras" --output_folder "imgs/synthetic/$cluster_num"
done

echo "All clusters processed."
