#!/bin/bash

# =========================================================================
# 実験: Med-EASi × <FKGL> 1B・易化事例のみで学習・エポック別評価（コピー崩壊の切り分け②：データ）
#   - 目的: フル版 train の 34.2% は参照文の FKGL が原文から下がっておらず、
#           「易しくしない」ことを教えている。これを除くとコピーが減るかを確かめる。
#   - データ: data/splits_flattened_full_fkgldown（src/build_fkgldown_split.py で生成）
#           train 1499→987・val 191→129（参照 FKGL < 原文 FKGL の事例のみ）。test はフル版と同じ 203 件。
#   - exp/fkgl-full-v2-epochs との差分はデータのみ（--save_each_epoch で 3 エポック学習し切り、
#     各エポック末の重みを全部評価する点も同じ）。エポック数をそろえて比較できる。
#     ※ train が少ないので 1 エポックあたりの更新回数は約 2/3（94→62 step）になる点に注意。
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
            --local_data_dir "data/splits_flattened_full_fkgldown" \
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
