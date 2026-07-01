#!/bin/bash

# Wrapper script for easy visualization generation
# Usage: ./visualize_comparison.sh <dataset> <control_attr> <model1> <model2> [model3] ...

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <dataset> <control_attr> <model1> <model2> [model3] ..."
    echo ""
    echo "Examples:"
    echo "  $0 Med-EASi FKGL Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct"
    echo "  $0 SimPA ARI model1 model2 model3"
    echo ""
    echo "Available datasets: Med-EASi, SimPA, WikiLarge_ori_splitwise, Newsela_s"
    echo "Available control attributes: FKGL, ARI, DALE-CHALL, CHAR_COMPRESSION, WORD_COMPRESSION"
    exit 1
fi

DATASET=$1
CONTROL_ATTR=$2
shift 2
MODELS=("$@")

# Default metrics - commonly used ones
METRICS=(
    "BLEU_to_source"
    "BLEU_to_ref"
    "BERTScore_to_source"
    "BERTScore_to_ref"
    "COMET"
    "SARI"
)

# Results file
RESULTS_FILE="output/sft_results/all_results.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file not found: $RESULTS_FILE"
    exit 1
fi

# Output directory
OUTPUT_DIR="output/nonsft_results_visualizations/${DATASET}_${CONTROL_ATTR}"

echo "========================================="
echo "Model Comparison Visualization"
echo "========================================="
echo "Dataset: $DATASET"
echo "Control Attribute: $CONTROL_ATTR"
echo "Models: ${MODELS[*]}"
echo "Metrics: ${METRICS[*]}"
echo "Output: $OUTPUT_DIR"
echo "========================================="

# Run Python visualization script
python src/visualize_model_comparison.py \
    --results "$RESULTS_FILE" \
    --dataset "$DATASET" \
    --control-attr "$CONTROL_ATTR" \
    --models "${MODELS[@]}" \
    --metrics "${METRICS[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --plot-types all

echo ""
echo "✓ Visualizations complete!"
echo "Check results in: $OUTPUT_DIR"
