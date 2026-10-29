#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <FKGL> 1B・目標値を書き換える実験（学習は exp/fkgl-full-v2-epochs と同一）
#   - 学習は exp/fkgl-full-v2-epochs（シード42・早期終了なし3エポック）をそのまま再現する
#     （サーバーの重みは掃除で消えているため学習し直す）。
#   - 推論側（run_sft_inference.sh）で、エポック3の重みに「出力側の FKGL タグ（目標）だけを書き換えた test」を与え、
#     参照文以外の目標にも従って出力の難しさが変わるか（介入による制御の確認）を確かめる。
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存
mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

MAX_LENGTH=512                      # 大語彙で巨大ロジット→OOM回避のため短めに
                                    # （KEEP版と同値。FKGL タグは数値1個でプロンプトはKEEP版より短く、切り捨てリスクはさらに低い）

# --- model name（本走行の 1B モデル）
#   ※ gated モデルのため、サーバー側で HF トークン（ライセンス承認済み）が必要
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

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
            --save_each_epoch \
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
