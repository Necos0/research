#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <FKGL>＋<KEEP> 同時制御 1B フルデータ版・エポック別評価（ロードマップ ④-4.1）
#   - 目的: 同じ平易化の制御（FKGL タグ）に KEEP タグを足すと、数値の保持率が上がるかを確かめる。
#           FKGL 単独モデルは目標 FKGL を下げるほど数値を落とす（目標4で 33/87 個、④-4.3）。
#           同時制御モデルで、低い目標でも数値が残るかを見る。
#   - タグは出力側 "<FKGL=y> <KEEP=a, b>"、原文側 "<FKGL=x> <KEEP=...>"（METRIC_NAME=FKGL_KEEP）。
#   - exp/fkgl-full-v2-epochs・exp/keep-full-v2-epochs との差分は METRIC_NAME と MAX_LENGTH のみ
#     （データ・ハイパラ・3エポック・早期終了なし・各エポック末の重みを評価、は同一）。
#   - シード 42。43・44 は従来どおり -seed43 / -seed44 ブランチで回す。
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存
mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

MAX_LENGTH=768                      # 単独モデル（512）より長くする。タグ2つ分の指示と説明で
                                    # プロンプトが KEEP 単独より約90トークン長く、512 だと train の
                                    # 約1割で簡約文の末尾が切れる（単独モデルでは約1.5%）。
                                    # 切れた件数は学習ログの "--- truncation:" で確認する

# --- model name（本走行の 1B モデル）
#   ※ gated モデルのため、サーバー側で HF トークン（ライセンス承認済み）が必要
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

DATASETS=(
    "medeasi"                       # ← ローカル folder 名（splits_flattened_filtered/medeasi）
    )
METRICS=(
    "FKGL_KEEP"
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
            --seed "42" \
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
