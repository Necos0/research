#!/bin/bash

# =========================================================================
# 実験: Med-EASi × 制御トークンなし（NOTAG）1B フルデータ版・エポック別評価（対照条件）
#   - 目的: KEEP は FKGL より数値を保持するが、数値あり事例で原文に近い（保守的）。
#           「数値の情報を与えたから保持できた」のか「KEEP タグは易しさを指示しないので
#           そもそも平易化が控えめになるだけ」なのかを切り分けるため、制御トークンを
#           一切使わない（数値の情報も易しさの情報も与えない）条件を学習する（ロードマップ③-9）。
#   - プロンプトは「INSTRUCTION: Simplify the following text. / SOURCE TEXT: 原文」のみ。
#     出力側にもタグを置かず、空の assistant ターンから簡約文を書かせる。
#   - exp/keep-full-v2-epochs との差分は METRIC_NAME=NOTAG のみ（データ・ハイパラ・シードは同一。
#     Trainer に seed を渡す修正込み、42 なら従来と同一）。
# =========================================================================

export WANDB_MODE=disabled          # W&B を使わない（著者entityへのログを回避）

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存
mkdir -p logs
LOG_FILE="logs/finetune_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

MAX_LENGTH=512                      # 大語彙で巨大ロジット→OOM回避のため短めに
                                    # （FKGL v2 と同値。比較の公平性のため変更しない）

# --- model name（本走行の 1B モデル）
#   ※ gated モデルのため、サーバー側で HF トークン（ライセンス承認済み）が必要
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

DATASETS=(
    "medeasi"                       # ← ローカル folder 名（splits_flattened_filtered/medeasi）
    )
METRICS=(
    "NOTAG"
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
