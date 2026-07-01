#!/bin/bash

# Navigate to your working directory
echo "current working directory:"
pwd

echo "Running on $(hostname)"
nvidia-smi

# Run your Python script
echo "Running the script ./src/SFT_prompting.py ..."
python ./src/SFT_prompting.py
