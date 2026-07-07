#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <KEEP> 最小GPUスモークテスト（ロードマップ2.1）
#   - 目的: <KEEP>（数値保持）制御トークンの学習パイプラインが端から端まで
#           動くことを、研究室GPUで最小サイズ・非gatedモデルで確認する。
#   - データはローカル data/splits_flattened_full/medeasi（フル・未フィルタ、keep付き）を
#     少量スライス使用。場所は python 呼び出しの --local_data_dir で指定する。
#   - ※事前に `python src/add_keep_metric.py --dataset medeasi` で
#     source_metrics/target_metrics に "keep" を付与しておくこと（本ブランチは付与済み）。
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存
mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

MAX_LENGTH=1024                     # Med-EASi は短文。最小化のため短めに

# --- model name（非gatedの小型モデルでスモーク。1B実走行は別ブランチ 2.2 で）
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

DATASETS=(
    "medeasi"                       # ← ローカル folder 名（splits_flattened_filtered/medeasi）
    )
METRICS=(
    "KEEP"
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
            --local_data_dir "data/splits_flattened_full" \
            --slice_train "32" \
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
