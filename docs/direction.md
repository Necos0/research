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
- 悪い点
    - 先行研究で行っていた平易化レベルの制御能力を捨てる。
    　→中間発表は、<keep>タグのみの影響を検証する。


# ロードマップ(中間発表まで)

- [ ] Medeasiのデータセット、`<FKGL>` タグを使った再現実験を行う
  - [x] 学習サイズを限りなく小さくし、研究室GPUでテスト
  - [ ] 1Bのモデルで実際に動かす
- [ ] Medeasiのデータセットに対して、`<keep>` タグで保持する情報を追加する
  - [ ] 学習サイズを限りなく小さくし、研究室GPUでテスト
  - [ ] 1Bのモデルで実際に動かす
- [ ] `<keep>` タグを使って1Bのモデルをファインチューニングする
- [ ] 結果を比較する

---

## 実験の実行フロー（共通運用）

- **実行環境**: 学習・推論・評価はすべて **研究室サーバー（CUDA GPU）** で回す。手元の Mac は CUDA 非搭載のため、**コード編集のみ**（学習は回さない）。
- **転送手段**: サーバーへの反映は **git ではなく rsync（SSH 経由の差分転送）** で行う。共有マシンに GitHub 認証情報を置かず、SSH ログインだけで済ませるため。Mac 側では従来どおり `exp/<実験名>` ブランチで **編集・コミットして履歴を残す**（サーバーへは push しない）。
- **配置**: リポジトリ・HF キャッシュ・conda env は、サーバーの割当領域 **`/mnt/gpu/workspace/2025/yuto_wada`** 配下に**すべて置く**（共有ストレージを圧迫しない）。
- **仮想環境**: `environment.yml` から **conda 仮想環境を構築** して実行する（conda-forge ベースで `cuda-toolkit`＋`pytorch` を env に同梱する研究室標準の流儀。定義は `taming-CATS/environment.yml`）。
- **バージョン管理**: 実験ごとに **ブランチを切って** 再現性を担保する。`main` は常に動く状態に保つ。

```
（Mac）編集・コミット → rsync で workspace へ送信
   →（サーバー）conda env 構築 → 実験実行 → rsync で結果を回収
```

### 手順

`<user>@<host>` は研究室サーバーに置き換える。リポジトリ・キャッシュ・env はすべて割当領域 `/mnt/gpu/workspace/2025/yuto_wada` 配下に置く。

1. **（Mac）実験用ブランチを作成**
   ```bash
   git switch -c exp/<実験名>        # 例: exp/fkgl-medeasi-repro
   ```
2. **（Mac）設定・コードを編集してコミット**（履歴を残すだけ。サーバーへは push しない）
   - 主な変更対象は `run_sft_finetune.sh` / `run_sft_inference.sh` の変数（`MODEL_NAME` / `DATASETS` / `METRICS` 等）。必要なら `src/`。
   - **README.md の「現在の実験」節に、このブランチの実験概要を記載する**（ブランチごとに必ず更新する）。最低限、次の3項目を書く：
     - **ブランチ**: `exp/<実験名>`
     - **目的**: この実験で確認・検証したいこと（1〜2行）
     - **設定**: モデル / データ / 制御属性 / データ件数・エポック・max_length / 推論条件 など、他条件と区別できる主要パラメータ
   ```bash
   git add -A && git commit -m "exp: <条件の説明>"
   ```
3. **（Mac）割当領域へ送信**（rsync。編集後の再送も同じコマンドで差分だけ転送）
   ```bash
   rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
     /Users/wadaketsunin/research <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/
   ```
4. **（サーバー）SSH して conda 仮想環境を有効化**
   ```bash
   ssh <user>@<host>
   cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS
   conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto   # HF_HOME / WANDB_MODE は env に登録済み
   # 依存を更新したら: conda env update -f environment.yml --prune
   # env を抜けるとき: conda deactivate
   ```
5. **（サーバー）実験を回す**（env を activate 済みなら HF_HOME / WANDB_MODE は設定済み）
   ```bash
   nvidia-smi                             # 空き GPU を確認
   export CUDA_VISIBLE_DEVICES=0          # 空いている番号を指定（実行するシェルごとに指定）
   ./run_sft_finetune.sh                  # → models/ に保存
   ./run_sft_inference.sh                 # → output/ に保存（末尾で評価も自動実行）
   ```
6. **（Mac）結果を回収**（rsync でサーバーから手元へ）
   ```bash
   rsync -av <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output/ \
     /Users/wadaketsunin/research/taming-CATS/output/
   ```
   - モデル重み（`*.safetensors`）は `.gitignore` 対象で重いので、必要な分だけ同様に rsync で回収する。評価サマリ（`output/sft_results/all_results.json`）や図を手元に取り込んで確認する。

### 共通の前提修正（作業ブランチに1度だけ）

- **データ読み込み**: 現状コードは `shtosti/<DATASET>`（著者の HF Hub）から取得する（`src/helpers/hugging_face.py`）。ローカル同梱の `data/splits_flattened_filtered/` を使うなら、**ローカル読み込み分岐を追加するパッチをベース作業ブランチに1度入れる**（`--dataset_name medeasi` で参照）。
- サーバーは CUDA のため `bf16=True` はそのままでよい（Mac 固有の dtype パッチは不要）。


