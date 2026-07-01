#!/bin/bash

DATASET_NAME="combined"


python ./src/dataset_stats.py \
  --splits_path ./data/splits_flattened_filtered/$DATASET_NAME \
  --save_dir ./data/splits_flattened_filtered/$DATASET_NAME/stats

echo "script executed successfully for dataset: $DATASET_NAME"