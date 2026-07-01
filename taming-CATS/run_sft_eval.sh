#!/bin/bash

echo "Script started: $(date)"

# TODO
DATA_DIR="WORD_COMPRESSION-WikiLarge_ori_splitwise-token_explanation-Llama-3.2-1B-Instruct-20250519"

# Derive metric and dataset from DATA_DIR prefix: METRIC-DATASET-...
METRIC_NAME=""
DATASET=""
REMAINDER=""

for metric_candidate in "FKGL" "ARI" "DALE-CHALL" "CHAR_COMPRESSION" "WORD_COMPRESSION"; do
  for dataset_candidate in "Med-EASi" "SimPA" "WikiLarge_ori_splitwise" "Newsela_s"; do
    prefix="${metric_candidate}-${dataset_candidate}-"
    if [[ "$DATA_DIR" == "$prefix"* ]]; then
      METRIC_NAME="$metric_candidate"
      DATASET="$dataset_candidate"
      REMAINDER="${DATA_DIR#$prefix}"
      break 2
    fi
  done
done

if [[ -z "$METRIC_NAME" || -z "$DATASET" || -z "$REMAINDER" ]]; then
  echo "[ERROR] Could not infer metric/dataset from DATA_DIR: $DATA_DIR"
  echo "[ERROR] Expected format: <METRIC>-<DATASET>-(<USER_PROMPT_ID>-<MODEL_NAME>|<MODEL_NAME>-<USER_PROMPT_ID>)-<YYYYMMDD>"
  echo "[ERROR] Supported metrics: FKGL, ARI, DALE-CHALL, CHAR_COMPRESSION, WORD_COMPRESSION"
  echo "[ERROR] Supported datasets: Med-EASi, SimPA, WikiLarge_ori_splitwise, Newsela_s"
  exit 1
fi

# Parse remaining segment: USER_PROMPT_ID-MODEL_NAME-YYYYMMDD
if [[ "$REMAINDER" =~ ^(.+)-([0-9]{8})$ ]]; then
  CORE_NO_DATE="${BASH_REMATCH[1]}"
  RUN_DATE="${BASH_REMATCH[2]}"
else
  echo "[ERROR] Could not parse date suffix from DATA_DIR remainder: $REMAINDER"
  echo "[ERROR] Expected trailing date format: YYYYMMDD"
  exit 1
fi

USER_PROMPT_ID=""
MODEL_NAME=""

# Support both conventions:
# 1) <USER_PROMPT_ID>-<MODEL_NAME>
# 2) <MODEL_NAME>-<USER_PROMPT_ID>
if [[ "$CORE_NO_DATE" =~ ^(token_[^-]+)-(.*)$ ]]; then
  USER_PROMPT_ID="${BASH_REMATCH[1]}"
  MODEL_NAME="${BASH_REMATCH[2]}"
elif [[ "$CORE_NO_DATE" =~ ^(.*)-(token_[^-]+)$ ]]; then
  MODEL_NAME="${BASH_REMATCH[1]}"
  USER_PROMPT_ID="${BASH_REMATCH[2]}"
else
  USER_PROMPT_ID="${CORE_NO_DATE%%-*}"
  MODEL_NAME="${CORE_NO_DATE#${USER_PROMPT_ID}-}"
fi

if [[ -z "$USER_PROMPT_ID" || -z "$MODEL_NAME" || "$MODEL_NAME" == "$CORE_NO_DATE" ]]; then
  echo "[ERROR] Could not infer USER_PROMPT_ID and MODEL_NAME from: $CORE_NO_DATE"
  exit 1
fi

BASE_DIR="output/sft_inference"
# BASE_DIR="output/baseline_inference"
INPUT_DIR="$BASE_DIR/$DATA_DIR"

shopt -s nullglob
INPUT_FILES=("$INPUT_DIR"/output_[0-9]*.json)
shopt -u nullglob

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
  echo "[ERROR] No input files found matching: $INPUT_DIR/output_[0-9]*.json"
  exit 1
fi

echo "Using DATASET=$DATASET"
echo "Using METRIC_NAME=$METRIC_NAME"
echo "Using USER_PROMPT_ID=$USER_PROMPT_ID"
echo "Using MODEL_NAME=$MODEL_NAME"
echo "Using RUN_DATE=$RUN_DATE"


python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/sft_results/all_results.json"

echo "Script completed: $(date)"