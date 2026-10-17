#!/bin/bash

# --- 標準出力＋標準エラーをタイムスタンプ付きログに保存（SARI/LENS/MAE 等が残る）
#   端末表示は tee で維持しつつ logs/ にも書き出す。logs/* は .gitignore 済み。
mkdir -p logs
LOG_FILE="logs/inference_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

echo "Script started: $(date)"

# =========================================================================
# 実験: Med-EASi × <FKGL> 1B 易化事例のみで学習・エポック別評価（各エポック末のチェックポイントを全件テスト・1シード）
export WANDB_MODE=disabled

# --- GPU メモリ対策（32GB GPU で LLM 生成と評価モデルが同居するため）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 断片化による OOM を回避
# SKIP_BERTSCORE / SKIP_LENS は export しない（export すると後段の sft_eval.py まで
# スキップされ、backfill が効かなくなる）。推論プロセスにのみ前置きで渡す。

METRIC_NAME="FKGL"
DATASET="medeasi"                       # ← ローカル folder 名
MODEL_NAME="Llama-3.2-1B-Instruct"      # short name（学習で使ったモデル）
USER_PROMPT_ID="token_explanation"
LOCAL_DATA_DIR="data/splits_flattened_full_fkgldown"   # 学習時と同じデータを指定すること（test はフル版と同じ 203 件）
# true: models/<MODEL_DIR>/checkpoint-*（学習の --save_each_epoch で残した各エポック末の重み）を全部評価する。
# false: 従来どおり models/<MODEL_DIR> 直下の最終（早期終了ならベスト）重みだけを評価する。
EVAL_EACH_CHECKPOINT=true

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
SEEDS=(
  37
  )

# 1つの重み（MODEL_PATH）について 推論 → 評価 → コピー率の集計 を行う
run_one() {
  local MODEL_PATH="$1" OUTPUT_DIR="$2" LABEL="$3"
  mkdir -p "$OUTPUT_DIR"
  echo
  echo "##### [$LABEL] MODEL_PATH=$MODEL_PATH → $OUTPUT_DIR"

  local i=1
  for SEED in "${SEEDS[@]}"; do
    echo "Running inference $i with seed $SEED..."
    local OUTPUT_FILE="$OUTPUT_DIR/output_$i.json"
    local ARGS=(
      # --use_vllm
      --seed "$SEED"
      --model_path "$MODEL_PATH"
      --dataset_name "$DATASET"
      --local_data_dir "$LOCAL_DATA_DIR"
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
    # SKIP_LENS / SKIP_BERTSCORE: 推論中は LENS・BERTScore を計算せず（roberta-large 等の
    # 評価モデルが生成中の LLM と GPU を取り合い OOM の主因になるため）、後段の
    # sft_eval.py の backfill で LLM 解放後にまとめて計算する（この推論プロセスのみに適用）
    SKIP_LENS=1 SKIP_BERTSCORE=1 python src/sft_inference.py "${ARGS[@]}" || {
      echo "ERROR: inference [$LABEL] (seed $SEED) が失敗しました。評価はスキップします。"
      exit 1
    }
    echo "Inference $i completed."
    ((i++))
  done

  echo "Running evaluation script [$LABEL]..."
  local INPUT_FILES=( "$OUTPUT_DIR"/output_*.json )   # 実際に生成されたシード分だけを集約
  python src/sft_eval.py \
    --input_files "${INPUT_FILES[@]}" \
    --metric_key "$METRIC_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --metric_mapping "data/metric_mapping.json"\
    --model_name "$MODEL_NAME"\
    --dataset "$DATASET"\
    --user_prompt_id="$USER_PROMPT_ID"\
    --summary_file="output/sft_results/all_results_${LABEL}.json"

  # 原文コピー率（SARI/LENS/MAE はコピー崩壊を検出できないため必ず併記する）
  python src/copy_stats.py --label "$LABEL" "$OUTPUT_DIR/output_1.json" | tee "$OUTPUT_DIR/copy_stats.txt"
}

if [ "$EVAL_EACH_CHECKPOINT" = true ]; then
  CKPTS=$(ls -1d "$MODELS_DIR/$MODEL_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
  if [ -z "$CKPTS" ]; then
    echo "ERROR: $MODELS_DIR/$MODEL_DIR/checkpoint-* がありません（学習を --save_each_epoch で回しましたか？）"
    exit 1
  fi
  for CKPT in $CKPTS; do
    # trainer_state.json の epoch（例 1.0, 2.0, 3.0）でラベル付けする
    EPOCH=$(python -c "import json,sys; print(round(json.load(open(sys.argv[1]))['epoch']))" "$CKPT/trainer_state.json")
    run_one "$CKPT" "output/sft_inference/$MODEL_DIR/epoch${EPOCH}" "epoch${EPOCH}"
  done
  echo
  echo "===== エポック別のコピー率（まとめ）"
  cat output/sft_inference/"$MODEL_DIR"/epoch*/copy_stats.txt
else
  run_one "$MODELS_DIR/$MODEL_DIR" "output/sft_inference/$MODEL_DIR" "final"
fi

echo "Script completed: $(date)"
