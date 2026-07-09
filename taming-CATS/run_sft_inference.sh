#!/bin/bash

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存（SARI/LENS/MAE 等が残る）
#   端末表示は tee で維持しつつ logs/ にも書き出す。logs/* は .gitignore 済み。
mkdir -p logs
LOG_FILE="logs/inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

echo "Script started: $(date)"

# =========================================================================
# 実験: Med-EASi × <KEEP> 1B モデルでの実走行（ロードマップ2.2・全件テスト・1シード）
export WANDB_MODE=disabled

# --- GPU メモリ対策（32GB GPU で LLM 生成と評価モデルが同居するため）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 断片化による OOM を回避
export SKIP_BERTSCORE=1   # BERTScore は roberta-large を毎回 GPU にロードして OOM の主因になるためスキップ（backfill なし・この実験では未使用）

METRIC_NAME="KEEP"
DATASET="medeasi"                       # ← ローカル folder 名
MODEL_NAME="Llama-3.2-1B-Instruct"      # short name（学習で使ったモデル）
USER_PROMPT_ID="token_explanation"

# 学習で生成された最新の models/ ディレクトリを自動選択
MODEL_DIR=$(ls -1dt "models/${METRIC_NAME}-${DATASET}-${USER_PROMPT_ID}-"* 2>/dev/null | head -1 | xargs -n1 basename)
if [ -z "$MODEL_DIR" ]; then
  echo "ERROR: models/${METRIC_NAME}-${DATASET}-${USER_PROMPT_ID}-* が見つかりません。先に ./run_sft_finetune.sh を実行してください。"
  exit 1
fi
echo "Using MODEL_DIR=$MODEL_DIR"
# =========================================================================


USE_PEFT=false
MODELS_DIR="models"
MODEL_PATH="$MODELS_DIR/$MODEL_DIR"
OUTPUT_DIR="output/sft_inference/$MODEL_DIR"
mkdir -p "$OUTPUT_DIR"

SEEDS=(
  37
  )
i=1
for SEED in "${SEEDS[@]}"; do
  echo "Running inference $i with seed $SEED..."

  OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"

  ARGS=(
    # --use_vllm
    --seed "$SEED"
    --model_path "$MODEL_PATH"
    --dataset_name "$DATASET"
    --local_data_dir "data/splits_flattened_full"   # 学習時と同じデータを指定すること
    --model_class "auto"
    --model_family "base"
    --max_length 1024
    --batch_size 8
    --slice_test -1
    --output_file "$OUTPUT_FILE"
    --control_tokens "data/prompts/control_tokens.json"
    --system_prompts "data/prompts/system_prompts.json"
    --user_prompts "data/prompts/user_prompts.json"
    --metric_mapping "data/metric_mapping.json"
    --metric_name "$METRIC_NAME"
    --user_prompt_id "$USER_PROMPT_ID"
  )

  if [ "$USE_PEFT" = true ]; then
    echo "Using PEFT..."
    ARGS+=( --peft_path "$MODEL_PATH" )
  fi

  # 推論が失敗したら即中断（出力ファイルが無いまま評価に進むのを防ぐ）
  # SKIP_LENS: 推論中は LENS を計算せず（LLM と GPU を取り合うため）、後段の
  # sft_eval.py の backfill でまとめて計算する（この推論プロセスのみに適用）
  SKIP_LENS=1 python src/sft_inference.py "${ARGS[@]}" || {
    echo "ERROR: inference (seed $SEED) が失敗しました。評価はスキップします。"
    exit 1
  }

  echo "Inference $i completed."
  ((i++))

done


# =========================================================================
echo "Running evaluation script..."

INPUT_DIR=$OUTPUT_DIR
INPUT_FILES=( "$INPUT_DIR"/output_*.json )   # 実際に生成されたシード分だけを集約

python src/sft_eval.py \
  --input_files "${INPUT_FILES[@]}" \
  --metric_key "$METRIC_NAME" \
  --output_dir "$INPUT_DIR" \
  --metric_mapping "data/metric_mapping.json"\
  --model_name "$MODEL_NAME"\
  --dataset "$DATASET"\
  --user_prompt_id="$USER_PROMPT_ID"\
  --summary_file="output/sft_results/all_results.json"

echo "Script completed: $(date)"