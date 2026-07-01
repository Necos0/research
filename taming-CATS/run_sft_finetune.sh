#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <FKGL> 再現（最小サイズのGPUスモークテスト）
#   - データはローカル data/splits_flattened_filtered/medeasi を使用
#   - 学習を極小化（16件・1エポック）してパイプライン疎通を確認する
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

MAX_LENGTH=1024                     # Med-EASi は短文。最小化のため 4096→1024

# --- model name（最小テストは 0.5B。本走行なら 1B などに切替）
#   ※ transformers==4.48.3 は Qwen3 未対応のため Qwen2.5 系を使う
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

DATASETS=(
    "medeasi"                       # ← ローカル folder 名（splits_flattened_filtered/medeasi）
    )
METRICS=(
    "FKGL"
    )

for DATASET_NAME in "${DATASETS[@]}"; do
    for METRIC_NAME in "${METRICS[@]}"; do
        echo "*** Finetuning $MODEL_NAME with $DATASET_NAME and $METRIC_NAME ***"

        CUDA_LAUNCH_BLOCKING=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TORCH_USE_CUDA_DSA=1 \
        python src/sft_finetune.py \
            --model_class "auto" \
            --model_family "base" \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME" \
            --slice_train "16" \
            --slice_val "8" \
            --batch_size "4" \
            --eval_batch_size "4" \
            --gradient_accumulation_steps "4" \
            --learning_rate "5e-6" \
            --weight_decay "0.01" \
            --warmup_steps "30" \
            --max_grad_norm "0.5" \
            --logging_steps "10" \
            --epochs "1" \
            --patience "4" \
            --max_length "$MAX_LENGTH"\
            --wandb_project_name "ATS_with_control_tokens" \
            --wandb_entity "shtosti"\
            --prompting_type "vanilla" \
            --user_prompt_id "token_explanation" \
            --metric_name "$METRIC_NAME" \
            --control_tokens "data/prompts/control_tokens.json" \
            --system_prompts "data/prompts/system_prompts.json" \
            --user_prompts "data/prompts/user_prompts.json"\
            --metric_mapping "data/metric_mapping.json"\
            --generate_every "20"\

    done
done