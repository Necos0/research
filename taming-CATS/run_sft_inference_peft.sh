#!/bin/bash

echo "Script started: $(date)"

# =========================================================================
# TODO
MODEL_DIR="CHAR_COMPRESSION-Newsela_s-token_explanation-Qwen3-8B-20250625"
CTRL_ATTR="CHAR_COMPRESSION"
DATASET="Newsela_s"
MODEL_NAME="Qwen3-8B"


# MODEL_NAME_HF="mistralai/Mistral-7B-Instruct-v0.1"
# MODEL_NAME_HF="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME_HF="meta-llama/Llama-2-13b-chat-hf"
MODEL_NAME_HF=Qwen/Qwen3-8B
# MODEL_NAME_HF=Qwen/Qwen3-14B


USER_PROMPT_ID="token_explanation"
# =========================================================================


MODELS_DIR="models"
MODEL_PATH="$MODELS_DIR/$MODEL_DIR"
OUTPUT_DIR="output/sft_inference/$MODEL_DIR"
mkdir -p "$OUTPUT_DIR"
USE_PEFT=true

SEEDS=(
  37 
  15 
  96 
  # 2 
  # 28
  )
i=1
for SEED in "${SEEDS[@]}"; do
  echo "Running inference $i with seed $SEED..."

  OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"

  ARGS=(
    --seed "$SEED"
    --model_path "$MODEL_PATH"
    --model_name "$MODEL_NAME_HF"
    --dataset_name "$DATASET"
    --model_class "auto"
    --model_family "base"
    --max_length 4096
    --batch_size 4
    --slice_test -1
    --output_file "$OUTPUT_FILE"
    --control_tokens "data/prompts/control_tokens.json"
    --system_prompts "data/prompts/system_prompts.json"
    --user_prompts "data/prompts/user_prompts.json"
    --metric_mapping "data/metric_mapping.json"
    --metric_name "$CTRL_ATTR"
    --user_prompt_id "$USER_PROMPT_ID"
  )

  if [ "$USE_PEFT" = true ]; then
    echo "Using PEFT..."
    ARGS+=( --peft_path "$MODEL_PATH" )
  fi

  python src/sft_inference.py "${ARGS[@]}"

  ((i++))
  echo "Inference $i completed."

done


# ==============================================================

echo "Running evaluation script..."

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
  --metric_key "$CTRL_ATTR" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/sft_results/all_results.json"

echo "Script completed: $(date)"