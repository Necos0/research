# 再現実験ガイド — どこを直せばよいか

このリポジトリは**著者のクラスタ環境（University of Zurich / SLURM / conda 環境 `sft`）固有のパスや
W&B 設定がハードコードされている**ため、そのままでは動かない箇所がある。
ここでは「再現実験のためにまず直すべき場所」を、優先度・段階別に整理する。

---

## 0. まず最初に直す（環境依存・全実験共通）

| 直す場所 | 現状 | 直し方 |
| --- | --- | --- |
| `Makefile` の `PYTHON` | `/home/hhubar/data/conda/envs/sft/bin/python`（著者環境固定） | 自分の環境の python パスに変更。または `Makefile` を使わず `run_*.sh` を直接実行 |
| W&B 設定（`run_sft_finetune.sh` 等の `--wandb_entity "shtosti"` / `--wandb_project_name`） | 著者の W&B アカウント | 自分の entity / project に変更。使わないなら学習コードの `wandb` ログを無効化（`WANDB_MODE=disabled` 環境変数が手軽） |
| HF アクセストークン | Llama/Mistral 等は gated | `huggingface-cli login` でトークン設定。`.env`（`python-dotenv`）にも対応 |
| `data/models.json` の Mistral パス | `Mistral-7B-Instruct-v0.1` の `path` が実は `v0.3` を指す等の不整合あり | 使うモデルの path を確認・修正 |
| 環境構築 | `environment.yml`（conda 名 `sft`, py3.12）/ `requirements.txt` | `conda env create -f environment.yml` か `pip install -r requirements.txt`。`vllm` は任意（`--use_vllm` 用） |
| GPU 前提 | bf16・CUDA 前提のコードあり | GPU 必須。小さいモデル（Llama-3.2-1B / Qwen3-1.7B / Ministral-3b）から始めると良い |

> SLURM が無い環境では `slurm_*.sh` / `slurm_zh/` は使わず、`run_*.sh`（ローカル用）を直接叩く。

---

## 1. データ準備を再現するなら

実際に学習が読むのは `data/splits_flattened_filtered/<dataset>/{train,val,test}.jsonl`。
**すでに成果物がコミットされていれば、データ準備はスキップして学習から始められる**（まず `ls data/splits_flattened_filtered/` を確認）。

ゼロから作り直す場合の段階と直す場所：

1. **生データ → JSONL**（`src/dataset_to_jsonl.py`）
   - `main()` 内で処理対象データセットがコメントアウトで切り替えられている。動かしたいものを有効化。
   - `classes/Dataset.py` の各ローダが**生データの置き場所を前提**にしている。手元の生データパスに合わせて修正が必要（生データ自体はリポジトリに含まれない可能性が高い → 各データセットを別途入手）。
   - BERTScore に `microsoft/deberta-xlarge-mnli` を使うため初回 DL が走る。
2. **分割**（`src/generate_splits.py`）
   - `main()` の定数を編集：`DATASETS`（対象）, `STRAT_METRIC="FKGL"`, `NUM_BINS=25`, `SEED=42`, `MAX_SAMPLES=3000`。
   - 外れ値除去は 1/99 パーセンタイル固定（関数 `remove_outliers_by_*`）。
3. **平坦化**（`src/flatten_splits.py` / `run_flatten_splits.sh`）
   - CLI: `--dataset`, `--keep_fraction`（既定 0.4 で間引き）, `--seed 42`。
4. **品質フィルタ**（`src/filter_flattened_splits.py`）
   - `filter_metrics(..., metrics=["FKGL","ARI","Dale-Chall"])` で「全指標が下がる例のみ残す」。判定指標を変えるならここ。`_hq` 版を生成。
5. **結合**（`src/create_combined_dataset.py`）
   - `main()` の `dataset_paths` で結合対象を指定（`seed=42` でシャッフル）。

> 再現性の要：乱数シードは全段階で `42` 固定。変えると分割・間引きが変わる。

---

## 2. SFT 学習を再現するなら（最頻出）

**入口は `run_sft_finetune.sh`。中の変数を書き換えるだけで主要な実験条件は変えられる。**

```bash
# run_sft_finetune.sh の編集ポイント
MODEL_NAME="Qwen/Qwen3-1.7B"     # ← 学習するベースモデル（models.json 参照）
DATASETS=( "Newsela_s" )          # ← 学習データ（_hq=フィルタ済 / combined=結合版 も選べる）
METRICS=( "CHAR_COMPRESSION" "WORD_COMPRESSION" )  # ← 制御する属性
```

`sft_finetune.py` に渡る主な引数と意味（ハイパラを変えるならここ）：

| 引数 | 既定（スクリプト内） | 意味 |
| --- | --- | --- |
| `--model_name` | スクリプトの `MODEL_NAME` | ベースモデル |
| `--model_family` | `base` | `llama`/`mistral`/`qwen`/`base`（chat template の扱い） |
| `--dataset_name` | ループ変数 | `data/splits_flattened_filtered/<dataset>` を読む |
| `--metric_name` | ループ変数 | 学習する制御属性（単一） |
| `--user_prompt_id` | `token_explanation` | `no_token`/`token`/`token_explanation`/`token_explanation_examples` |
| `--learning_rate` | `5e-6` | フル FT の学習率 |
| `--batch_size` / `--gradient_accumulation_steps` | `4` / `4` | 実効バッチ＝16 |
| `--epochs` / `--patience` | `3` / `4` | エポック数・早期終了 |
| `--max_length` | `4096` | 最大系列長（VRAM に直結） |
| `--max_grad_norm` `--weight_decay` `--warmup_steps` | `0.5` `0.01` `30` | 最適化 |
| `--seed` | `42` | 乱数シード |

**典型的な変更パターン**
- 別モデルで試す → `MODEL_NAME` と `--model_family` を変更。
- 別データ/別属性 → `DATASETS` / `METRICS` 配列を編集（複数指定すると総当たりでループ）。
- プロンプト変種の比較（アブレーション）→ `--user_prompt_id` を切替。
- **VRAM 不足** → `--max_length` を下げる / `--batch_size` を下げる / **PEFT 版**（`run_sft_finetune_peft.sh`＋`--peft --lora_r`）に切替 / より小さいモデルにする。
- **複数属性を 1 モデルで** → `run_sft_finetune_all_ctrl_attr.sh`（`sft_finetune_all_ctrl_attr.py`, `--metric_names` 複数）を使う。

学習結果は `models/<METRIC>-<DATASET>-token_explanation-<MODEL>-<日付>/` に保存される。

> プロンプトの中身そのものを変えたい場合は `data/prompts/*.json` を編集し、
> 組み立てロジックは `src/helpers/prompting.py` を直す。

---

## 3. 推論を再現するなら

**入口は `run_sft_inference.sh`。** 冒頭の `TODO` ブロックを学習済みモデルに合わせて書き換える：

```bash
MODEL_DIR="FKGL-Med-EASi-Ministral-3b-instruct-token_explanation-20250518"  # ← models/ 配下のディレクトリ名
METRIC_NAME="FKGL"      # ← モデルが学習した属性と一致させる
DATASET="Med-EASi"      # ← 評価に使うテストセット
MODEL_NAME="Ministral-3b-instruct"
USER_PROMPT_ID="token_explanation"
```

- `SEEDS=(37 15 96 2 28)`：複数シードで推論し、評価で集約する（再現性の中核）。減らせば速い。
- `sft_inference.py` の主な引数：`--temperature 0.0`（貪欲法＝既定）, `--batch_size`, `--max_length 4096`,
  `--slice_test -1`（全件 / 数値で部分実行＝デバッグ用）, `--use_vllm`（高速化、要 vllm）。
- PEFT モデルなら `USE_PEFT=true` にすると `--peft_path` が付く。
- このスクリプトは**推論後に自動で `sft_eval.py`** を呼び `output/sft_results/all_results.json` に追記する。

出力は `output/sft_inference/<MODEL_DIR>/output_<i>.json`。

**ベースライン（未学習モデル）と比較するなら** `run_baseline_inference.sh` を使う
（`MODEL_PATH=MODEL_NAME` とし、HF から直接ロード。結果は `output/baseline_inference/` と `output/nonsft_results_baseline/`）。

---

## 4. 評価だけやり直すなら

推論済みの `output_*.json` があるなら `run_sft_eval.sh` で評価のみ再実行できる。

```bash
DATA_DIR="WORD_COMPRESSION-WikiLarge_ori_splitwise-token_explanation-Llama-3.2-1B-Instruct-20250519"
# ↑ output/sft_inference/ 配下のディレクトリ名。metric/dataset/prompt/model はこの名前から自動逆算される
```

- `sft_eval.py` が要求値 vs 達成値の **Pearson 相関 / MAE / RMSE / 範囲内割合** を計算し、散布図を出力。
- `--metric_key` は計測する制御属性。`--metric_mapping data/metric_mapping.json` で属性名→項目名を解決。
- サマリは `--summary_file`（既定 `output/sft_results/all_results.json`）に蓄積される。
- 複数設定を比較するなら `run_sft_eval_compare.sh` / `run_sft_eval_batch.sh`。

---

## 5. 図を作り直すなら

| スクリプト | 用途 | 出力 |
| --- | --- | --- |
| `visualize_comparison.sh` → `visualize_simplification_results.py` | モデル/設定間の比較プロット | `visualizations/` |
| `visualize_control_errors.sh` → `visualize_control_errors.py` | 制御誤差（要求値からのズレ）の可視化 | `visualizations/control_errors/` |
| `src/analyse_results.ipynb` | 結果の対話的分析 | — |

---

## 最短の再現フロー（おすすめ順）

VRAM が限られた環境を想定した、現実的な最小構成：

1. **環境を作る**（§0）：conda 環境 + HF login + `WANDB_MODE=disabled`。
2. **学習データの有無を確認**：`ls data/splits_flattened_filtered/`。
   あればデータ準備（§1）はスキップ。無ければ §1 を実施。
3. **小モデルで 1 条件だけ学習**：`run_sft_finetune.sh` を
   `MODEL_NAME=Qwen/Qwen3-1.7B`（または Llama-3.2-1B）, `DATASETS=("Med-EASi_hq")`, `METRICS=("FKGL")` にして実行。
4. **推論＋評価**：`run_sft_inference.sh` の `MODEL_DIR` を生成された `models/` のディレクトリ名にし、
   `SEEDS` を 1〜2 個に減らして実行 → `output/sft_results/all_results.json` を確認。
5. **ベースライン比較**（任意）：`run_baseline_inference.sh` で未学習版を同条件で評価。
6. **図**（任意）：§5。
7. うまくいったら、モデル・データセット・属性・プロンプト変種を増やして本実験へ。

---

## つまずきやすいポイント

- **`MODEL_DIR` 名のパース**：`run_sft_eval.sh` は `<METRIC>-<DATASET>-...-<YYYYMMDD>` の命名からメタ情報を逆算する。
  自前で名前を変えるとパースに失敗する → 命名規則を守るか、`sft_eval.py` を直接引数指定で呼ぶ。
- **metric 名の表記ゆれ**：CLI では `DALE-CHALL`、データ項目は `Dale-Chall`。変換は `data/metric_mapping.json` 経由。
- **WikiLarge の "splitwise/global/ori"**：原 split は保持されていない。`documents/wikilarge_processing_history.md` を必読。
- **生データ非同梱**：`data/datasets/*/dataset.jsonl` 以前の生データはリポジトリに無いことが多い。§1 を完全再現するには各データセットを別途入手。
- **シード**：データ準備は `42` 固定、推論は複数シードで集約。ここを変えると数値が一致しない。
- **`documents/` は理論、本 docs は実装手順**：手法の根拠は `taming-CATS/documents/` の各 md に詳しい。
