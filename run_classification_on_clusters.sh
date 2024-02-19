#!/bin/bash

for i in {1..10}
do
   printf -v dataset_name "%02d" $i  # Format dataset name with leading zero
   echo "Running for dataset part $dataset_name"
   python classification.py --tfrecord_path "data/train-val-test/clusters/diabetic-retinopath-128-16-labeled_train_part_$i.tfrecord" --dataset_name "$dataset_name"
done
