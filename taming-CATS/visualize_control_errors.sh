#!/bin/bash
# Wrapper script for visualizing control attribute prediction errors

# Default values
BASELINE_RESULTS="output/nonsft_results_baseline/all_results.json"
FINETUNED_RESULTS="output/sft_results/all_results.json"
OUTPUT_DIR="visualizations/control_errors"
COMPARE_MODE=true  # By default, compare baseline vs finetuned

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> <control_attr1> [control_attr2 ...] [--models model1 model2 ...]"
    echo ""
    echo "Example:"
    echo "  $0 Med-EASi FKGL ARI CHAR_COMPRESSION  # Auto-detect models, multiple control attrs"
    echo "  $0 Med-EASi FKGL --models Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct"
    echo "  $0 Med-EASi FKGL  # Auto-detect all models, single control attr"
    echo ""
    echo "Note: Model names should match keys in all_results.json (without org prefix)"
    echo "      By default, compares baseline vs finetuned models"
    exit 1
fi

DATASET=$1
shift

# Collect control attributes until we hit --models or end of args
CONTROL_ATTRS=()
while [ $# -gt 0 ] && [ "$1" != "--models" ]; do
    CONTROL_ATTRS+=("$1")
    shift
done

# If we hit --models, collect the models
if [ "$1" == "--models" ]; then
    shift
    MODELS=("$@")
else
    MODELS=()
fi

# If no models specified, try to find them automatically from all_results.json
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models specified. Extracting available models from $BASELINE_RESULTS..."
    
    # Use the first control attr to find models
    FIRST_CTRL_ATTR="${CONTROL_ATTRS[0]}"
    
    # Use python to extract model names for this dataset/control_attr
    MODELS_FOUND=($(python3 -c "
import json
with open('$BASELINE_RESULTS', 'r') as f:
    data = json.load(f)
models = []
for model in data.keys():
    if '$DATASET' in data[model] and '$FIRST_CTRL_ATTR' in data[model]['$DATASET']:
        models.append(model)
print(' '.join(models))
" 2>/dev/null))
    
    if [ ${#MODELS_FOUND[@]} -eq 0 ]; then
        echo "Error: No models found for $DATASET - $FIRST_CTRL_ATTR in $BASELINE_RESULTS"
        echo "Please specify models manually."
        exit 1
    fi
    
    echo "Found ${#MODELS_FOUND[@]} models:"
    for model in "${MODELS_FOUND[@]}"; do
        echo "  - $model"
    done
    
    MODELS=("${MODELS_FOUND[@]}")
fi

# Activate virtual environment
source venv/bin/activate

# Run visualization
echo ""
echo "Generating control error visualizations..."
echo "Dataset: $DATASET"
echo "Control Attributes: ${CONTROL_ATTRS[@]}"
echo "Models: ${MODELS[@]}"
echo ""

if [ "$COMPARE_MODE" = true ]; then
    python src/visualize_control_errors.py \
        --baseline_results "$BASELINE_RESULTS" \
        --finetuned_results "$FINETUNED_RESULTS" \
        --dataset "$DATASET" \
        --control_attrs "${CONTROL_ATTRS[@]}" \
        --models "${MODELS[@]}" \
        --output_dir "$OUTPUT_DIR"
    
    echo ""
    echo "Visualizations saved to: $OUTPUT_DIR"
    echo "  - Bar charts: ${DATASET}_<CTRL_ATTR>_comparison_MAE.png"
    echo "  - Scatter canvas (aggregated): ${DATASET}_scatter_canvas.png"
    echo "  - Per-sample scatter: ${DATASET}_per_sample_scatter.png"
else
    python src/visualize_control_errors.py \
        --baseline_results "$BASELINE_RESULTS" \
        --dataset "$DATASET" \
        --control_attrs "${CONTROL_ATTRS[@]}" \
        --models "${MODELS[@]}" \
        --output_dir "$OUTPUT_DIR"
    
    echo ""
    echo "Visualizations saved to: $OUTPUT_DIR"
    echo "  - Bar charts: ${DATASET}_<CTRL_ATTR>_MAE.png"
fi
