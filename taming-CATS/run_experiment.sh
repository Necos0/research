#!/bin/bash
# =========================================================================
# サーバーでの実験を1コマンドで回す。
#
#   ./run_experiment.sh <ブランチ名> [GPU番号]
#
# 例:
#   ./run_experiment.sh exp/keep-medeasi-1b-v2        # GPU は自動選択
#   ./run_experiment.sh exp/fkgl-medeasi-1b-full-v2 1 # GPU 1 を明示
#
# やること（この順で全部）:
#   1. ブランチを fetch/switch/pull し、そのブランチ版の本スクリプトで実行し直す
#   2. tmux セッションを自動で張り、その中で以降を実行（SSH が切れても継続）
#   3. conda env を有効化・HF_TOKEN を確認・空き GPU を選択
#   4. 前実験の output/models/logs と **HF datasets キャッシュ** を掃除
#      （キャッシュを残すと修正前のプロンプトで学習してしまう。実際に事故った）
#   5. 学習 → 推論を連続実行（学習が失敗したら推論に進まない）
#   6. Mac へ結果を回収する scp コマンドを表示
#
# 事前に一度だけ: HF トークンを置いておく（gated モデル用）
#   echo '<your_hf_token>' > ~/.hf_token && chmod 600 ~/.hf_token
# =========================================================================
set -euo pipefail

REPO="/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS"
CONDA_ENV="/mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto"

BRANCH="${1:-}"
GPU_ARG="${2:-auto}"

if [ -z "$BRANCH" ]; then
    echo "使い方: ./run_experiment.sh <ブランチ名> [GPU番号]"
    echo "例:     ./run_experiment.sh exp/keep-medeasi-1b-v2"
    exit 1
fi

cd "$REPO"

# --- STAGE 1: ブランチを引く -------------------------------------------------
# 引いた後は「そのブランチ版の」本スクリプトで実行し直す（自己更新）。
if [ "${_STAGE:-0}" -lt 1 ]; then
    echo "=== [1/6] ブランチを取得: $BRANCH"
    git fetch origin
    git switch "$BRANCH"
    git pull --ff-only
    export _STAGE=1
    exec "$REPO/run_experiment.sh" "$@"
fi

# --- STAGE 2: tmux の中に入る ------------------------------------------------
# 学習は長時間かかるため、SSH が切れても死なないよう tmux 内で回す。
SESSION="exp-$(echo "$BRANCH" | sed 's#^exp/##; s#[^A-Za-z0-9_-]#-#g')"
if [ -z "${TMUX:-}" ]; then
    echo "=== [2/6] tmux セッション '$SESSION' を作成して実行"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "ERROR: セッション '$SESSION' が既に存在します。"
        echo "  進捗を見る:   tmux attach -t $SESSION"
        echo "  消してやり直す: tmux kill-session -t $SESSION"
        exit 1
    fi
    exec tmux new-session -s "$SESSION" \
        "_STAGE=2 '$REPO/run_experiment.sh' $(printf '%q ' "$@"); echo; echo '--- 終了しました。Enter でセッションを閉じます'; read"
fi

# --- STAGE 3: 実行環境の準備 -------------------------------------------------
echo "=== [3/6] 実行環境を準備"

# conda activate はシェル関数なので、非対話シェルではフック(conda.sh)を読む必要がある。
# 既に関数として使えるなら何もしない（declare -F で判定）。
if ! declare -F conda >/dev/null 2>&1; then
    CONDA_BASE=""
    command -v conda >/dev/null 2>&1 && CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    for c in "$CONDA_BASE" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /opt/miniconda3; do
        if [ -n "$c" ] && [ -f "$c/etc/profile.d/conda.sh" ]; then
            source "$c/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! declare -F conda >/dev/null 2>&1; then
    echo "ERROR: conda を activate できません（conda.sh が見つからない）。"
    echo "  手動で conda activate '$CONDA_ENV' してから run_sft_finetune.sh を実行してください。"
    exit 1
fi
# conda の activate スクリプトは未定義変数（PS1 等）を参照するため、ここだけ set -u を外す
set +u
conda activate "$CONDA_ENV"
set -u
echo "  conda env : ${CONDA_PREFIX:-?}"

# HF トークン（gated モデル: meta-llama/Llama-3.2-1B-Instruct）
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.hf_token" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.hf_token")"
    export HF_TOKEN
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN が未設定です（gated モデルの取得に必要）。"
    echo "  echo '<your_hf_token>' > ~/.hf_token && chmod 600 ~/.hf_token"
    exit 1
fi
echo "  HF_TOKEN  : 設定済み"

# GPU（未指定なら使用メモリが最小のものを選ぶ）
if [ "$GPU_ARG" = "auto" ]; then
    GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
    echo "  GPU       : $GPU （空きが最大のものを自動選択）"
else
    GPU="$GPU_ARG"
    echo "  GPU       : $GPU （指定）"
fi
export CUDA_VISIBLE_DEVICES="$GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | sed 's/^/    /'

# --- STAGE 4: 掃除 -----------------------------------------------------------
echo "=== [4/6] 前実験の生成物と datasets キャッシュを掃除"
for d in output models logs; do
    if [ -d "$d" ]; then
        echo "  削除対象: $d/ ($(du -sh "$d" 2>/dev/null | cut -f1))"
    fi
done
if [ -d output ] || [ -d models ]; then
    echo "  ※ これらは前実験の結果です。Mac 側 results/<実験名>/ に回収済みか確認してください。"
    read -r -p "  削除して続行しますか？ [y/N] " ans
    case "$ans" in
        [yY]*) ;;
        *) echo "中止しました（何も削除していません）。"; exit 1 ;;
    esac
fi
rm -rf output models logs

# datasets のキャッシュ。これを残すと map のフィンガープリントが変わらない場合に
# **修正前のプロンプトで学習してしまう**（FKGL v2 / KEEP v2 がこれで壊れた）。
# コード側でも load_from_cache_file=False にしてあるが、二重に潰しておく。
HF_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$HF_CACHE_ROOT/datasets" ]; then
    echo "  削除: $HF_CACHE_ROOT/datasets"
    rm -rf "$HF_CACHE_ROOT/datasets"
fi
# load_dataset がローカル JSONL の隣に吐く cache-*.arrow も消す
find "$REPO/data" \( -name "cache-*.arrow" -o -type d -name "cache-*" \) -exec rm -rf {} + 2>/dev/null || true

# --- STAGE 5: 学習 → 推論 ----------------------------------------------------
echo "=== [5/6] 学習を開始（ブランチ: $BRANCH / GPU: $GPU）"
echo "    ※ 開始直後に 'prompt sanity OK: ...<TAG=...>' が出ることを確認してください。"
echo "       出なければプロンプト構築が壊れています（assert で落ちます）。"
echo
START_TS=$(date +%s)

./run_sft_finetune.sh
echo
echo "=== 学習が完了。推論を開始"
./run_sft_inference.sh

ELAPSED=$(( ($(date +%s) - START_TS) / 60 ))

# --- STAGE 6: 後始末の案内 ---------------------------------------------------
EXP_NAME="${BRANCH#exp/}"
echo
echo "=== [6/6] 完了（所要 ${ELAPSED} 分）"
echo
echo "Mac 側で以下を実行して結果を回収してください:"
echo
echo "  mkdir -p /Users/wadaketsunin/research/results/${EXP_NAME}"
echo "  scp -r wada_yuto@calc40:${REPO}/output \\"
echo "    wada_yuto@calc40:${REPO}/models \\"
echo "    wada_yuto@calc40:${REPO}/logs \\"
echo "    /Users/wadaketsunin/research/results/${EXP_NAME}/"
echo
echo "回収後、このセッションを消す:  tmux kill-session -t ${SESSION}"
