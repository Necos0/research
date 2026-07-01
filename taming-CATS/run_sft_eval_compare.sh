#!/bin/bash

# =========================================================================
# TODO
DATASETS=(
  "Newsela_s" 
  # "Newsela" 
  "Med-EASi" 
  # "Med-EASi_hq" 
  # "NoFluff" 
  # "NoFluff_hq" 
  "WikiLarge_ori_splitwise" 
  # "WikiLarge_ori_splitwise_hq"
  "SimPA" 
  # "SimPA_hq"
  )
CTRL_ATTRS=(
  "FKGL" 
  "ARI" 
  "DALE-CHALL" 
  "CHAR_COMPRESSION" 
  "WORD_COMPRESSION"
  )
# =========================================================================

SAVE_DIR="output/sft_results"
mkdir -p "$SAVE_DIR"

# Run the evaluation script for each dataset and control attribute
for DATASET in "${DATASETS[@]}"; do
  for CTRL_ATTR in "${CTRL_ATTRS[@]}"; do
    echo "Running evaluation for dataset: $DATASET, control attribute: $CTRL_ATTR"
    
    python src/sft_eval_compare.py \
      --summary_file="output/sft_results/all_results.json" \
      --dataset="$DATASET" \
      --control_attr="$CTRL_ATTR" \
      --save_dir="$SAVE_DIR" \
      --color_map_path="data/colormap/color_map.json"\
      --user_prompt_id="token_explanation"
      
    echo "Evaluation completed for dataset: $DATASET, control attribute: $CTRL_ATTR"
  done
done