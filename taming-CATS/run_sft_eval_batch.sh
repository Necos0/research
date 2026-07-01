#!/bin/bash

set -euo pipefail

echo "Script started: $(date)"

# Optional first argument: parent directory containing all experiment folders.
BASE_DIR="${1:-output/sft_inference}"
SUMMARY_FILE="output/sft_results/all_results.json"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "[ERROR] Base directory does not exist: $BASE_DIR"
  exit 1
fi

infer_from_data_dir() {
  local data_dir="$1"
  local metric_name=""
  local dataset=""
  local remainder=""

  # for metric_candidate in "FKGL" "ARI" "DALE-CHALL" "CHAR_COMPRESSION" "WORD_COMPRESSION"; do
  for metric_candidate in "FKGL" "CHAR_COMPRESSION" ; do
    # for dataset_candidate in "Med-EASi" "SimPA" "WikiLarge_ori_splitwise" "Newsela_s"; do
    for dataset_candidate in "Med-EASi_hq" "SimPA_hq" "WikiLarge_ori_splitwise_hq" "Newsela_s_hq"; do
      local prefix="${metric_candidate}-${dataset_candidate}-"
      if [[ "$data_dir" == "$prefix"* ]]; then
        metric_name="$metric_candidate"
        dataset="$dataset_candidate"
        remainder="${data_dir#$prefix}"
        break 2
      fi
    done
  done

  if [[ -z "$metric_name" || -z "$dataset" || -z "$remainder" ]]; then
    return 1
  fi

  if [[ "$remainder" =~ ^(.+)-([0-9]{8})$ ]]; then
    local core_no_date="${BASH_REMATCH[1]}"
    local run_date="${BASH_REMATCH[2]}"
    local user_prompt_id=""
    local model_name=""

    # Support both conventions:
    # 1) <USER_PROMPT_ID>-<MODEL_NAME>
    # 2) <MODEL_NAME>-<USER_PROMPT_ID>
    if [[ "$core_no_date" =~ ^(token_[^-]+)-(.*)$ ]]; then
      user_prompt_id="${BASH_REMATCH[1]}"
      model_name="${BASH_REMATCH[2]}"
    elif [[ "$core_no_date" =~ ^(.*)-(token_[^-]+)$ ]]; then
      model_name="${BASH_REMATCH[1]}"
      user_prompt_id="${BASH_REMATCH[2]}"
    else
      user_prompt_id="${core_no_date%%-*}"
      model_name="${core_no_date#${user_prompt_id}-}"
    fi

    if [[ -z "$user_prompt_id" || -z "$model_name" || "$model_name" == "$core_no_date" ]]; then
      return 1
    fi

    echo "$metric_name|$dataset|$user_prompt_id|$model_name|$run_date"
    return 0
  fi

  return 1
}

shopt -s nullglob
DATA_DIR_PATHS=("$BASE_DIR"/*/)
shopt -u nullglob

if [[ ${#DATA_DIR_PATHS[@]} -eq 0 ]]; then
  echo "[ERROR] No experiment directories found under: $BASE_DIR"
  exit 1
fi

ok_count=0
skip_count=0
fail_count=0

for input_dir_path in "${DATA_DIR_PATHS[@]}"; do
  INPUT_DIR="${input_dir_path%/}"
  DATA_DIR="$(basename "$INPUT_DIR")"

  if ! parsed="$(infer_from_data_dir "$DATA_DIR")"; then
    echo "[SKIP] Unrecognized folder format: $DATA_DIR"
    ((skip_count+=1))
    continue
  fi

  IFS='|' read -r METRIC_NAME DATASET USER_PROMPT_ID MODEL_NAME RUN_DATE <<< "$parsed"

  shopt -s nullglob
  INPUT_FILES=("$INPUT_DIR"/output_[0-9]*.json)
  shopt -u nullglob

  if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "[SKIP] No input files in: $INPUT_DIR"
    ((skip_count+=1))
    continue
  fi

  echo "--------------------------------------------------"
  echo "Running eval for DATA_DIR=$DATA_DIR"
  echo "DATASET=$DATASET | METRIC_NAME=$METRIC_NAME | USER_PROMPT_ID=$USER_PROMPT_ID | MODEL_NAME=$MODEL_NAME | RUN_DATE=$RUN_DATE"

  if python src/sft_eval.py \
    --input_files "${INPUT_FILES[@]}" \
    --metric_key "$METRIC_NAME" \
    --output_dir "$INPUT_DIR" \
    --metric_mapping "data/metric_mapping.json" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --user_prompt_id "$USER_PROMPT_ID" \
    --summary_file "$SUMMARY_FILE"; then
    ((ok_count+=1))
  else
    echo "[FAIL] Evaluation failed for: $DATA_DIR"
    ((fail_count+=1))
  fi
done

echo "--------------------------------------------------"
echo "Batch eval complete. Success=$ok_count, Skipped=$skip_count, Failed=$fail_count"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi

echo "Script completed: $(date)"
