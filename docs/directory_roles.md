# ディレクトリ・主要ファイルの役割

`taming-CATS/` 直下の構成と、それぞれが何を担うかをまとめる。
（パス表記はすべて `taming-CATS/` を起点とする。）

## トップレベルの全体像

| パス | 種別 | 役割 |
| --- | --- | --- |
| `src/` | コード | 全処理の本体（データ準備・学習・推論・評価・可視化のスクリプト群） |
| `data/` | 入力データ | 生データ→JSONL→分割→平坦化→フィルタ済みの各段階の成果物、プロンプト定義、設定 JSON |
| `models/` | 成果物 | SFT 済みモデルの保存先（`<METRIC>-<DATASET>-<MODEL>-<PROMPT>-<日付>/`） |
| `output/` | 成果物 | 推論結果（生成文＋計測値）と評価サマリ |
| `visualizations/` | 成果物 | 評価から生成した図（散布図・MAE 比較など） |
| `experiments/` | 成果物 | 分割サンプリング／層化パラメータ探索の中間結果・プロット |
| `logs/` | ログ | クラスタ（SLURM）実行ログ。中身を読む必要は基本ない |
| `documents/` | ドキュメント | 著者による設計ノート（手法の理論的説明。本 docs の元ネタ） |
| `slurm_zh/` | 実行スクリプト | SLURM（クラスタ）投入用シェルスクリプト群 |
| `run_*.sh` | 実行スクリプト | ローカル実行用のエントリポイント（`src/*.py` を引数付きで呼ぶ） |
| `slurm_*.sh` | 実行スクリプト | トップ階層の SLURM 投入スクリプト |
| `Makefile` | 実行スクリプト | `make ft / infer / eval` などで SLURM ジョブを投げるショートカット |
| `environment.yml` / `requirements.txt` | 環境 | conda / pip 依存定義（`sft` という conda 環境想定） |
| `sweep.yaml` / `sweep_peft.yaml` | 設定 | W&B ハイパーパラメータスイープ定義 |
| `data/models.json` `data/metric_mapping.json` | 設定 | 利用可能モデル一覧、制御属性名→データセット項目名の対応 |

---

## `src/` — 処理本体

役割ごとにグルーピングして説明する。

### (A) データ準備パイプライン

| ファイル | 役割 | 主な設定箇所 |
| --- | --- | --- |
| `dataset_to_jsonl.py` | 各生データセットを統一 JSONL スキーマに変換し、可読性・圧縮率・BLEU/BERTScore を計算 | `main()` 内で処理対象データセットをコメントアウトで選択（既定 Newsela） |
| `classes/Dataset.py` | データセット別ローダ（`Newsela`/`MedEASi`/`WikiLarge`/`SimPALex`/`SimPASyn`）。生データの読み込みと整形 | クラスごとの入力パス・整形ロジック |
| `classes/Metrics.py` | 全可読性指標（FKGL/ARI/Dale-Chall/FRE）・文字/語/文カウント・圧縮率・BLEU・BERTScore を計算 | BERTScore は `microsoft/deberta-xlarge-mnli` |
| `generate_splits.py` | 外れ値除去（1/99 パーセンタイル）→ FKGL で 25 ビン層化 → 80/10/10 分割 | `main()` の `DATASETS` `STRAT_METRIC` `NUM_BINS` `SEED` `MAX_SAMPLES` |
| `flatten_splits.py` | 「1ソース＝複数簡約」を「1行＝1簡約」に平坦化。`--keep_fraction` で間引き | CLI 引数 `--dataset --keep_fraction --seed` |
| `filter_flattened_splits.py` | FKGL/ARI/Dale-Chall が**全て**下がる例だけ残す品質フィルタ（`_hq` 版生成）。Venn 図も出力 | `filter_metrics(..., metrics=[...])` の判定指標 |
| `create_combined_dataset.py` | 複数データセットの train/val/test をそれぞれ結合・シャッフルして `combined` 作成 | `main()` 内の `dataset_paths` |
| `create_simpa_subset.py` `create_combined_dataset.py` | SimPA 語彙/統語サブセットのマージ・結合 | — |

### (B) 分割サンプリング／層化パラメータ探索（実験用・本筋ではない）

| ファイル | 役割 |
| --- | --- |
| `find_best_stratification_parameters.py` / `..._wiki.py` | 層化のビン数や指標を変えて分布のズレ（JSD/EMD/KS）を最小化する設定を探索 |
| `experiment_with_splits.py` `explore_wikilarge_sampling.py` `generate_wikilarge_subset.py` | WikiLarge のサブサンプリング戦略比較 |
| `plot_wikilarge_divergencies*.py` | 上記の発散指標プロット |
| `explore_datasets*.py` `dataset_stats.py` | データセットの統計・分布の確認 |
| `generate_random_seeds.py` | 推論で使う乱数シード列（`data/random_seeds_*.txt`）生成 |

### (C) 学習（SFT）

| ファイル | 役割 | 主な設定 |
| --- | --- | --- |
| `sft_finetune.py` | **単一制御属性**の SFT 学習本体。プロンプト組み立て・ラベルマスク・学習ループ | CLI 引数（`--model_name --dataset_name --metric_name --user_prompt_id` 等） |
| `sft_finetune_all_ctrl_attr.py` | **複数制御属性**を 1 モデルで学習（metric をラウンドロビンで割当） | `--metric_names` 複数指定 |
| `sft_finetune_peft.py` 系 | LoRA/PEFT による省メモリ学習版（`run_sft_finetune_peft.sh` / `sweep_peft.yaml`） | `--peft --lora_r --lora_dropout` |
| `helpers/prompting.py` | プロンプト組み立ての中核：system プロンプト抽選、user プロンプト生成、説明・例の挿入、chat template 適用、completion 整形 | プロンプトロジックを変えるならここ |
| `helpers/hugging_face.py` | モデル/トークナイザ読み込み、HF への push 等 | — |
| `helpers/utils.py` | JSONL 入出力などの共通ユーティリティ | — |
| `classes/PredictionLoggerCallback.py` | 学習中に一定ステップごとに生成例をログ（W&B） | `--generate_every` |

### (D) 推論

| ファイル | 役割 | 主な設定 |
| --- | --- | --- |
| `sft_inference.py` | 学習済み（またはベースライン）モデルにテストセットを与え、指定した制御値で簡約文を生成し計測 | `--model_path --metric_name --temperature --use_vllm --slice_test` |
| `generate_responses.py` `generate_dynamic_responses.py` | データ準備段階での応答生成（`run_full_dataset_preparation.sh` から呼ばれる） | — |
| `WANDB_generate_predictions.py` | W&B 連携の予測生成 | — |

### (E) 評価・可視化

| ファイル | 役割 | 出力先 |
| --- | --- | --- |
| `sft_eval.py` | 複数シードの推論結果を集約し、要求値 vs 達成値の Pearson 相関 / MAE / RMSE / 範囲内割合を計算。散布図も生成 | `--output_dir` と `--summary_file`（`output/sft_results/all_results.json`） |
| `sft_eval_compare.py` | 複数モデル/設定の評価比較 | — |
| `classes/Metrics.py` | 評価でも生成文の指標計算に再利用 | — |
| `visualize_simplification_results.py` | 簡約結果の可視化 | `visualizations/` |
| `visualize_control_errors.py` | 制御誤差の可視化（要求値からのズレ） | `visualizations/control_errors/` |
| `analyse_results.ipynb` | 結果の対話的分析ノートブック | — |

### (F) その他

- `SFT_prompting.py`: プロンプティング実験。
- `check_load_from_hf.py` `explore_datasets_from_hf.py`: HF からのロード確認。
- `slurm_test.py`: クラスタ動作テスト。

---

## `data/` — データと設定

| パス | 役割 |
| --- | --- |
| `datasets/<name>/dataset.jsonl` | Stage 1：統一スキーマ JSONL（1ソース＋複数簡約の階層形式）。`medeasi/ newsela/ simpa/ wikilarge*/` など |
| `splits/` `splits_new/` | Stage 2：層化 80/10/10 分割後の `train/val/test.jsonl`＋`log.txt`。`splits_new/<name>_<MAX_SAMPLES>/` 形式 |
| `splits_flattened/` | Stage 3：1行1簡約に平坦化したもの |
| `splits_flattened_filtered/` | Stage 4：品質フィルタ後。**学習が実際に読むのはここ**。`combined/` は結合版、`stats.json` `venn.png` `bars.png` も同居 |
| `prompts/control_tokens.json` | 各制御トークンの説明文と段階的な例（`token_explanation*` 用） |
| `prompts/system_prompts.json` | system プロンプト 6 バリアント（学習時にランダム抽選） |
| `prompts/user_prompts.json` | metric × variant ごとの user プロンプトテンプレート |
| `prompts/old_prompts/` | 旧版プロンプト |
| `dataset_schema/v1〜v5.json` | JSONL スキーマのバージョン定義 |
| `metric_mapping.json` | 制御属性名（`DALE-CHALL` 等）→ データセット項目名（`Dale-Chall` 等）の対応表 |
| `models.json` | 利用可能モデル一覧（family / path / size / url） |
| `random_seeds_5.txt` `random_seeds_10.txt` | 推論で使うシード列 |
| `colormap/` | プロット用カラーマップ |

> WikiLarge の `splitwise` / `global` / `ori` の用語は紛らわしい。
> 詳細は `taming-CATS/documents/wikilarge_processing_history.md` を必ず参照（原 split は保持されておらず、約2,000件に間引き済みのサブセットから新規分割している）。

---

## `models/` — 学習済みモデル

- 命名規則は概ね `<METRIC>-<DATASET>-token_explanation-<MODEL>-<YYYYMMDD>/`
  （順序は実行スクリプトにより前後あり。`sft_eval.sh` はこの名前から metric/dataset/prompt/model を逆算している）。
- 各ディレクトリに重み・トークナイザ・学習設定が入る。
- 推論時は `--model_path models/<dir>` で指定。

## `output/` — 推論・評価成果物

| パス | 役割 |
| --- | --- |
| `sft_inference/<実験名>/output_<i>.json` | SFT モデルの推論結果（シードごと）。生成文＋計測値 |
| `baseline_inference/<model>-<dataset>-<metric>/` | 未学習（ベースライン）モデルの推論結果 |
| `sft_results/all_results.json` | SFT 評価の集約サマリ |
| `nonsft_results_baseline/all_results.json` | ベースライン評価の集約サマリ |
| `old_experiments/` | 旧実験の保管 |

## 実行スクリプト（`run_*.sh` / `slurm_*` / `Makefile`）

ローカル用 `run_*.sh` は対応する `src/*.py` を引数付きで呼ぶ薄いラッパ。
中の変数（`MODEL_NAME` / `DATASETS` / `METRICS` / `MODEL_DIR` など）を書き換えて使う。

| スクリプト | 呼ぶ処理 |
| --- | --- |
| `run_full_dataset_preparation.sh` | データ準備一式（`run_full_dataset_preparation.py`＋`generate_responses.py`） |
| `run_flatten_splits.sh` | `flatten_splits.py`（`--keep_fraction --dataset`） |
| `run_dataset_stats.sh` | `dataset_stats.py` |
| `run_sft_finetune.sh` / `_2.sh` / `_all_ctrl_attr.sh` / `_peft.sh` | 学習（モデル×データセット×metric をループ） |
| `run_sft_inference.sh` / `_1.sh` / `_peft.sh` | 推論（複数シード）→ 末尾で評価も実行 |
| `run_baseline_inference.sh` | 未学習モデルでの推論＋評価 |
| `run_sft_eval.sh` / `_batch.sh` / `_compare.sh` | 評価のみ（推論済み output を集約） |
| `visualize_comparison.sh` / `visualize_control_errors.sh` | 可視化 |
| `Makefile`（`make ft/infer/eval`） | 対応する `slurm_*.sh` を `sbatch` 投入 |
| `slurm_zh/*.sh`, `slurm_*.sh` | 上記をクラスタ（SLURM）で実行する版 |

> `Makefile` の `PYTHON` パスや各スクリプトの `wandb_entity` 等は著者のクラスタ環境固有。
> ローカル再現では要書き換え（[reproduction_guide.md](reproduction_guide.md) 参照）。
