#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <KEEP> 1B モデルでの実走行（ロードマップ2.2）
#   - 目的: <KEEP>（数値保持）制御トークンを、実運用サイズの 1B モデル・
#           フルデータ全件で学習し、数値保持の制御効果を測る。
#     設定は FKGL 1B 実走行（exp/fkgl-medeasi-1b）に合わせる
#     （全件・batch 4・grad_accum 4・lr 5e-6・3エポック・max_length 512）。
#   - データはローカル data/splits_flattened_full/medeasi（フル・未フィルタ、keep付き）を
#     全件使用。場所は python 呼び出しの --local_data_dir で指定する。
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存
mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

MAX_LENGTH=512                      # 大語彙で巨大ロジット→OOM回避のため短めに
                                    # （512超のプロンプトは train 1499件中1件・keepタグなし事例のみ。切り捨て影響なし）

# --- model name（本走行の 1B モデル）
#   ※ gated モデルのため、サーバー側で HF トークン（ライセンス承認済み）が必要
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

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
            --slice_train "-1" \
            --slice_val "-1" \
            --batch_size "4" \
            --eval_batch_size "4" \
            --gradient_accumulation_steps "4" \
            --learning_rate "5e-6" \
            --weight_decay "0.01" \
            --warmup_steps "30" \
            --max_grad_norm "0.5" \
            --logging_steps "10" \
            --epochs "3" \
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
