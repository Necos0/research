#!/bin/bash

echo "Script started: $(date)"

python src/generate_random_seeds.py \
    --n 5\
    --output_dir "./data"

echo "Script finished: $(date)"