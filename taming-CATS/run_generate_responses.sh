#!/bin/bash

PYTHON_SCRIPT="./src/generate_responses.py"

# Check if the script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

# Run the script
echo "Running the script $PYTHON_SCRIPT ..."
python "$PYTHON_SCRIPT"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Python script ran successfully!"
else
    echo "Error: Python script failed to run."
    exit 1
fi
