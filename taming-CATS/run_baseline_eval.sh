#!/bin/bash

echo "Script started: $(date)"

# TODO
DATA_DIR="Llama-3.2-1B-Instruct-WikiLarge_ori_splitwise-FKGL"
MODEL_NAME="Llama-3.2-1B-Instruct"

# Derive metric from DATA_DIR suffix
if [[ "$DATA_DIR" == *"-CHAR_COMPRESSION" ]]; then
  METRIC_NAME="CHAR_COMPRESSION"
  DATA_DIR_NO_METRIC="${DATA_DIR%-CHAR_COMPRESSION}"
elif [[ "$DATA_DIR" == *"-FKGL" ]]; then
  METRIC_NAME="FKGL"
  DATA_DIR_NO_METRIC="${DATA_DIR%-FKGL}"
else
  echo "[ERROR] Could not infer metric from DATA_DIR: $DATA_DIR"
  echo "[ERROR] Expected suffix: -FKGL or -CHAR_COMPRESSION"
  exit 1
fi

# Derive dataset from DATA_DIR (supported datasets only)
DATASET=""
for candidate in "Med-EASi" "SimPA" "WikiLarge_ori_splitwise" "Newsela_s"; do
  if [[ "$DATA_DIR_NO_METRIC" == *"-$candidate" ]]; then
    DATASET="$candidate"
    break
  fi
done

if [[ -z "$DATASET" ]]; then
  echo "[ERROR] Could not infer dataset from DATA_DIR: $DATA_DIR"
  echo "[ERROR] Supported datasets: Med-EASi, SimPA, WikiLarge_ori_splitwise, Newsela_s"
  exit 1
fi

# BASE_DIR="output/sft_inference"
BASE_DIR="output/baseline_inference"
INPUT_DIR="$BASE_DIR/$DATA_DIR"

shopt -s nullglob
INPUT_FILES=("$INPUT_DIR"/output_[0-9]*.json)
shopt -u nullglob

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No input files found matching: $INPUT_DIR/output_[0-9]*.json"
  exit 1
fi

echo "Using DATASET=$DATASET and METRIC_NAME=$METRIC_NAME"


python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="token_explanation"\
  --summary_file="output/nonsft_results_baseline/all_results.json"

echo "Script completed: $(date)"