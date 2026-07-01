#!/bin/bash

MODEL_PATH="sft/Llama-3.2-1B-Instruct-Med-EASi-20250418-1416-t227y2fq

python src/WANDB_generate_predictions.py \
  --model_path "$MODEL_PATH"" \
  --tokenizer_path huggingface/llama2-tokenizer \
  --dataset_name my-dataset-name \
  --split validation \
  --metric_name simplicity \
  --peft \
  --output_file ./predictions.json
