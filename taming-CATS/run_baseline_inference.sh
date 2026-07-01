#!/bin/bash

echo "Script started: $(date)"

# Download required NLTK data
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# =========================================================================
# BASELINE CONFIGURATION - Non-finetuned model from Hugging Face
# =========================================================================
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="Qwen/Qwen3-1.7B"
# MODEL_NAME="Qwen/Qwen3-8B"
# MODEL_NAME="ministral/Ministral-3b-instruct"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"

USER_PROMPT_ID="token_explanation"
TEMPERATURE=0.0

# Define datasets and metrics to iterate over
DATASETS=(
  # "Med-EASi"
  # "SimPA"
  "WikiLarge_ori_splitwise"
  # "Newsela_s"
)

METRICS=(
  # "ARI"
  # "FKGL"
  # "DALE-CHALL"
  "CHAR_COMPRESSION"
  # "WORD_COMPRESSION"
)

SEEDS=(
  37 
  15
  96 
  # 2 
  # 28
  )
# =========================================================================

# For baseline, model_path = model_name (no local finetuned model)
MODEL_PATH="$MODEL_NAME"

# Iterate over datasets and metrics
for DATASET in "${DATASETS[@]}"; do
  for METRIC_NAME in "${METRICS[@]}"; do
    echo ""
    echo "========================================================================="
    echo "Running baseline inference for DATASET=$DATASET, METRIC=$METRIC_NAME"
    echo "========================================================================="
    
    OUTPUT_DIR="output/baseline_inference/$(basename $MODEL_NAME)-$DATASET-$METRIC_NAME"
    mkdir -p "$OUTPUT_DIR"

    # Run inference with multiple seeds
    i=1
    for SEED in "${SEEDS[@]}"; do
      OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"
      
      # Skip if output already exists
      if [ -f "$OUTPUT_FILE" ]; then
        echo "Output $i already exists, skipping (file: $OUTPUT_FILE)"
        ((i++))
        continue
      fi
      
      echo "Running baseline inference $i with seed $SEED..."

      ARGS=(
        # --use_vllm  # Uncomment for faster inference if vllm is installed
        --seed "$SEED"
        --model_path "$MODEL_PATH"
        --model_name "$MODEL_NAME"
        --dataset_name "$DATASET"
        --model_class "auto"
        --model_family "base"  # Options: "llama", "mistral", "qwen", "base"
        --max_length 8192
        --batch_size 4
        --slice_test -1  # -1 for full test set, or specify a number for subset
        --output_file "$OUTPUT_FILE"
        --control_tokens "data/prompts/control_tokens.json"
        --system_prompts "data/prompts/system_prompts.json"
        --user_prompts "data/prompts/user_prompts.json"
        --metric_mapping "data/metric_mapping.json"
        --metric_name "$METRIC_NAME"
        --user_prompt_id "$USER_PROMPT_ID"
        --temperature "$TEMPERATURE"
      )

      # Note: No --peft_path for baseline (non-finetuned models)

      python src/sft_inference.py "${ARGS[@]}"

      ((i++))
      echo "Baseline inference $i completed."

    done

    # =========================================================================
    echo "Running evaluation script for $DATASET - $METRIC_NAME..."

    INPUT_DIR=$OUTPUT_DIR
    INPUT_FILES=(
      "$INPUT_DIR/output_1.json"
      "$INPUT_DIR/output_2.json"
      "$INPUT_DIR/output_3.json"
      # "$INPUT_DIR/output_4.json"
      # "$INPUT_DIR/output_5.json"
    )

    python src/sft_eval.py \
      --input_files "${INPUT_FILES[@]}" \
      --metric_key "$METRIC_NAME" \
      --output_dir "$INPUT_DIR" \
      --metric_mapping "data/metric_mapping.json"\
      --model_name "$(basename $MODEL_NAME)"\
      --dataset "$DATASET"\
      --user_prompt_id="$USER_PROMPT_ID"\
      --summary_file="output/nonsft_results_baseline/all_results.json"

    echo "Evaluation completed for $DATASET - $METRIC_NAME"
    echo ""
  done
done

echo "Script completed: $(date)"
