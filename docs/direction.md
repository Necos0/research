# 制御トークンを用いた数値情報の保持に基づくテキスト平易化

## 背景・目的
### 社会的意義
大規模言語モデル（LLM）を用いたテキスト平易化は、子ども、日本語学習者、知的・発達障害を持つ人々への情報アクセシビリティを向上させる上で極めて重要である。
### 入出力関係
原文→LLM→平易化文

## 課題
- 平易化レベルの制御
    - 読み手の属性（子ども、外国人学習者、障害の特性など）によって、必要とされる平易化の性質（語彙の平易化、構文の簡素化など）が異なる。
- ハルシネーション
    - 文章を平易化する過程で、原文の意味が歪められたり、内容が不当に圧縮されて重要な情報が消失したりするリスクがある。

## 既存研究
- ACCESS(2020)
    - 言語学的な特徴量（文字数、構文の深さ等）を制御タグとして入力に付与することで、1つのモデルで平易化レベルを多角的に調整する手法を確立した。
- Taming-CATS(2026)
    - ACCESSの思想を現代のLLMに適用。欧米で汎用的に使われる可読性指標（FKGL等）を制御トークンとして指示微調整（Instruction Fine-Tuning）に組み込み、専門ドメイン（医療・行政等）における精密なレベル制御を実現した。

## 既存研究の課題
既存研究は「読みやすさ（可読性）」や「文章の短さ（圧縮率）」の制御を優先するあまり、文脈を成立させるために不可欠なコア情報まで削ぎ落としてしまう（不当な情報欠落）という問題に対策ができていない。

## 提案手法
- ハ本研究では、ハルシネーションの中でも特に「数値情報の欠落」に焦点を当てる
    - 実用上の重要性：医療や行政などのドメインにおいて、数値（投薬量、年齢制限、確率など）の欠落や変形は、ユーザーの生命や権利に関わる致命的な問題となるため。
    - 自動検知・評価の容易性：主観的な意味の歪み（ハルシネーション）とは異なり、数値の脱落や変化は正規表現等のアルゴリズムを用いて人間を介さずに機械的に検知・抽出できるため。

具体的なアプローチ
Taming-CATSの「制御トークン」の仕組みを応用・拡張する。
学習時に、原文に含まれる重要な数値を抽出。原文の先頭に、消失させてはならない数値を明示的に指定する <keep=value> 形式の制御トークンを付与する。モデルの訓練を通して、「指定された数値情報を必ず出力に保持する」という指示をLLMに学習させる。

## トレードオフ
- 良い点
    - 変数が一個なため、因果がクリーンで、実装が最も軽い。
    - 数値情報を保持できる。
- 悪い点
    - 先行研究で行っていた平易化レベルの制御能力を捨てる。
    　→中間発表は、<keep>タグのみの影響を検証する。
    - 数値情報の保持によって既存の評価指標が低く出る可能性がある。


---

## 実験の実行フロー（共通運用）

- **実行環境**: 学習・推論・評価はすべて **研究室サーバー（CUDA GPU）** で回す。手元の Mac は CUDA 非搭載のため、**コード編集のみ**（学習は回さない）。
- **コードの受け渡し**: サーバーへのコード反映は **GitHub 経由の `git clone` / `git pull`** で行う。Mac 側で `exp/<実験名>` ブランチを **編集・コミットして GitHub へ push**、サーバー側で取り込む。公開リポジトリなので、サーバーは **HTTPS で読み取りのみ**（認証情報を共有マシンに置かない）。
- **ブランチ＝実験、clone は1個**: サーバーには clone を **1個だけ**置き、実験ごとに **`git fetch origin && git switch exp/<実験名>`**（同一ブランチ更新時は `git pull`）で **対象ブランチを引いて切り替える**。各ブランチが各実験に対応する。
- **切り替え時の注意**: `output/`・`models/`・`logs/` は `.gitignore` 対象で **ブランチを切り替えても消えない**。前実験の生成物が残ると `run_sft_inference.sh` が別実験のモデルを拾う恐れがあり、`logs/` も前実験のログが混ざったまま scp されてしまうため、**回収済みの `output/`・`models/`・`logs/` は削除してから**新しい実験を回す。
- **結果の回収**: 出力・実行ログは git に載らないため、サーバー → Mac へ **rsync で回収**する。Mac 側には `output/`・`models/`・`logs/` を置かず、実験別に **`results/<実験名>/`**（`results/` は `.gitignore` 済み）へ集約し、実験間で上書きしないようにする。
- **モデル重み（`*.safetensors`）は回収しない**（`--exclude='*.safetensors'`）。1実験あたり約 4.9GB あるがアーカイブの99%以上を占める一方、**Mac には CUDA が無く使い道がない**。再現性は「ブランチ＝実験」（コード・データ・シードが git にある）で担保されるので、重みを持つ必要はない。`models/` の `args.json` / `config.json` / `tokenizer.json` は小さいので**残す**（実験条件の証跡になり、tokenizer はローカルでのプロンプト再現に実際に役立つ）。
  - **例外**: 学習し直さずに**推論だけやり直す**可能性がある場合は、その実験の `models/` を**サーバーに残したままにする**（次の実験を回すと `run_experiment.sh` が消す）。重みを Mac に持ってきても、結局サーバーに戻さないと推論できない。
- **配置**: リポジトリ・HF キャッシュ・conda env は、サーバーの割当領域 **`/mnt/gpu/workspace/2025/yuto_wada`** 配下に**すべて置く**（共有ストレージを圧迫しない）。
- **仮想環境**: `environment.yml` から **conda 仮想環境を構築** して実行する（conda-forge ベースで `cuda-toolkit`＋`pytorch` を env に同梱する研究室標準の流儀。定義は `taming-CATS/environment.yml`）。
- **バージョン管理**: 実験ごとに **ブランチを切って** 再現性を担保する。`main` は常に動く状態に保つ。

```
（Mac）編集・コミット・push → GitHub
   →（サーバー）git clone/pull → conda env → 実験実行 → scp で結果を回収
```

### 手順

研究室サーバーは `wada_yuto@calc40`。リポジトリ・キャッシュ・env はすべて割当領域 `/mnt/gpu/workspace/2025/yuto_wada` 配下に置く。

1. **（Mac）実験用ブランチを作成**
   ```bash
   git switch -c exp/<実験名>        # 例: exp/fkgl-medeasi-repro
   ```
2. **（Mac）設定・コードを編集してコミット**（履歴を残すだけ。サーバーへは push しない）
   - 主な変更対象は `run_sft_finetune.sh` / `run_sft_inference.sh` の変数（`MODEL_NAME` / `DATASETS` / `METRICS` 等）。必要なら `src/`。
   - **README.md の「現在の実験」節に、このブランチの実験概要を記載する**（ブランチごとに必ず更新する）。最低限、次の3項目を書く：
     - **ブランチ**: `exp/<実験名>`
     - **目的**: この実験で確認・検証したいこと（1〜2行）
     - **設定**: 他条件と区別できる主要パラメータを列挙する。最低限、モデル / データ / 制御属性 / データ件数（train・val・test）/ **バッチサイズ**（＋ gradient accumulation）/ **学習率** / **エポック数** / max_length / 推論条件（シード等）。`run_sft_finetune.sh` / `run_sft_inference.sh` の実値と一致させる。
   ```bash
   git add -A && git commit -m "exp: <条件の説明>"
   ```
3. **（Mac）GitHub へ push**
   ```bash
   git push -u origin exp/<実験名>
   ```
4. **（サーバー）SSH → `run_experiment.sh` を1コマンド実行**

   ブランチ取得・tmux 作成・conda 有効化・掃除・学習・推論を**すべてこのスクリプトがやる**。手で `export` するものは無い（`HF_TOKEN` / `HF_HOME` / `WANDB_MODE` は **conda env の activate 時に設定済み**）。GPU も**サーバー側で割り当てられる**ので `CUDA_VISIBLE_DEVICES` の指定は不要。

   ```bash
   ssh wada_yuto@calc40
   cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS \
     && git fetch origin && git switch exp/<実験名> && git pull --ff-only \
     && ./run_experiment.sh exp/<実験名>
   ```

   - **先頭の `git fetch/switch/pull` は必ず付ける。** サーバーのローカルブランチが古いと `run_experiment.sh` 自体がまだ存在せず `No such file or directory` になる（スクリプトはブランチを切り替える側なので、自分自身を先に持ってこられない）。この3つを前置すれば、どんな状態からでも動く。
   - 初回だけ clone が必要: `cd /mnt/gpu/workspace/2025/yuto_wada && git clone https://github.com/Necos0/research.git`（公開リポジトリなので認証不要）
   - clone は割当領域に **1個だけ**。実験ごとに**ブランチを切り替える**（各ブランチ＝各実験）。
   - 特定の GPU に固定したいときだけ第2引数で指定: `./run_experiment.sh exp/<実験名> 1`
   - 依存を更新したら: `conda env update -f environment.yml --prune`

   スクリプトが自動でやること:

   | 段階 | 内容 |
   | --- | --- |
   | 1 | `git fetch` / `switch` / `pull` し、**そのブランチ版のスクリプトで実行し直す**（自己更新） |
   | 2 | **tmux セッションを自動作成**（`exp-<ブランチ名>`）。SSH が切れてもジョブは継続 |
   | 3 | conda env を有効化 |
   | 4 | 前実験の `output/` `models/` `logs/` と **HF datasets キャッシュ**を掃除（削除前に確認を求める） |
   | 5 | **学習 → 推論**を連続実行（学習が失敗したら推論に進まない） |
   | 6 | Mac へ結果を回収する scp コマンドを表示 |

   - **datasets キャッシュの削除は必須**。残すと `map` のフィンガープリントが変わらない場合に**修正前のプロンプトで学習してしまう**（実際に FKGL v2 / KEEP v2 がこれで壊れた）。コード側でも `load_from_cache_file=False` にしてあるが、二重に潰している。
   - **学習開始直後に `--- prompt sanity OK: ...'<TAG=...>'` が出ることを必ず確認する。** 出ない／assert で落ちる場合はプロンプト構築が壊れている。
   - 実行が始まったら **`Ctrl-b` → `d` で detach** し、SSH を切ってよい。
   - 進捗確認は再 SSH して `tmux attach -t exp-<ブランチ名>`。一覧は `tmux ls`、終わったら `tmux kill-session -t exp-<ブランチ名>`。

5. **（手動で回したい場合のみ）**
   ```bash
   tmux new -s exp-<実験名>
   conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto
   rm -rf output models logs "$HF_HOME/datasets"   # 掃除（回収済みが前提）
   ./run_sft_finetune.sh && ./run_sft_inference.sh
   ```
6. **（Mac）結果を回収**（rsync でサーバーから手元の `results/<実験名>/` へまとめる。**モデル重みは除外**）
   ```bash
   mkdir -p /Users/wadaketsunin/research/results/<実験名>
   rsync -av --exclude='*.safetensors' \
     wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output \
     wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/models \
     wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/logs \
     /Users/wadaketsunin/research/results/<実験名>/
   ```
   - **Mac 側には `output/`・`models/`・`logs/` を置かず、実験ごとに `results/<実験名>/` に集約する**（`results/` は `.gitignore` 済み。実験間で上書きされない）。
   - `--exclude='*.safetensors'` により、1実験あたりのアーカイブは **約4.9GB → 約20MB** になる。理由は上の「モデル重みは回収しない」を参照。

### 共通の前提修正（作業ブランチに1度だけ）

- **データ読み込み**: 現状コードは `shtosti/<DATASET>`（著者の HF Hub）から取得する（`src/helpers/hugging_face.py`）。ローカル同梱の `data/splits_flattened_filtered/` を使うなら、**ローカル読み込み分岐を追加するパッチをベース作業ブランチに1度入れる**（`--dataset_name medeasi` で参照）。
- サーバーは CUDA のため `bf16=True` はそのままでよい（Mac 固有の dtype パッチは不要）。


