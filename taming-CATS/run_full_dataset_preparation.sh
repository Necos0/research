#!/bin/bash

################## turn dataset to the json schema -> jsonl format ##################
PYTHON_SCRIPT="./src/run_full_dataset_preparation.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

echo "Running the script $PYTHON_SCRIPT ..."
python "$PYTHON_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Python script ran successfully!"
else
    echo "Error: Python script failed to run."
    exit 1
fi


################## ##################
PYTHON_SCRIPT="./src/generate_responses.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

echo "Running the script $PYTHON_SCRIPT ..."
python "$PYTHON_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Python script ran successfully!"
else
    echo "Error: Python script failed to run."
    exit 1
fi