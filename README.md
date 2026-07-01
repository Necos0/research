# 数値情報を保持する制御トークンによるテキスト平易化

LLM によるテキスト平易化において、可読性レベルの制御に加えて **「原文中の数値情報を欠落させない」** ことを制御トークンで実現する研究のリポジトリ。
先行研究 **Taming-CATS** の制御トークン機構を土台にする。

## 研究の狙い

- 既存手法（ACCESS / Taming-CATS）は可読性・圧縮率の制御を優先するあまり、**文脈に不可欠なコア情報（特に数値）まで削ぎ落とす**問題に未対応。
- 本研究は、原文の重要な数値を `<keep=value>` 制御トークンで明示し、**出力に必ず保持させる**ことを学習させる。
- 数値の欠落・改変は正規表現で機械的に検知でき、評価が客観的。
- **中間発表では `<keep>` タグ単体の効果**（変数を1つに絞ったアブレーション）を検証する。

詳細は [docs/direction.md](docs/direction.md) を参照。

## リポジトリ構成

| パス | 内容 |
| --- | --- |
| `taming-CATS/` | 実装本体（Taming-CATS 再現コード＋本研究の実験設定）。使用データは Med-EASi のみに整理済み |
| `docs/` | 研究方針・コード解説・再現ガイド（[docs/README.md](docs/README.md) が索引） |
| `2604.01779v1.pdf` | 参照論文 |

## ロードマップ（中間発表まで）

1. **Med-EASi × `<FKGL>` の再現実験** ← いまここ
   - 学習サイズを極小化して研究室GPUでスモークテスト
   - 1B モデルで実走
2. Med-EASi に `<keep>` タグで保持情報を追加
3. `<keep>` タグで 1B モデルをファインチューニング
4. `<keep>` あり/なしの結果比較

## 実験の実行フロー

- 学習・推論・評価は **研究室サーバー（CUDA GPU）** で実行。手元の Mac は編集用。
- サーバーの割当領域 **`/mnt/gpu/workspace/2025/yuto_wada`** 配下に、リポジトリ・HF キャッシュ・conda env を**すべて置く**（共有ストレージを圧迫しない／共有マシンに GitHub 認証情報を置かない）。
- 転送は **rsync**（GitHub 認証は不要、SSH ログインのみ）。Mac 側では従来どおり `exp/...` ブランチで編集・コミットして履歴を残す。

```
（Mac）編集・コミット → rsync で workspace へ送信
   →（サーバー）conda env 構築 → 実験実行 → rsync で結果を回収
```

conda 仮想環境は `taming-CATS/environment.yml`（conda-forge ベースで `cuda-toolkit`＋`pytorch` を同梱）から構築する。詳しい方針は [docs/direction.md](docs/direction.md)。

## 現在の実験：Med-EASi × `<FKGL>` 再現（最小GPUスモークテスト）

- **ブランチ**: `exp/fkgl-medeasi-repro`
- **目的**: データ準備→学習→推論→評価のパイプラインが端から端まで通ることを確認する（数値の再現ではなく疎通確認）。
- **設定**: モデル `Qwen/Qwen3-0.6B` / データ `medeasi`（ローカル）/ 制御属性 `FKGL` / train 16件・val 8件・1エポック・max_length 1024 / 推論は1シード・test 8件。

### 研究室サーバーでの実行

`<user>@<host>` は研究室サーバーに置き換える。リポジトリ・キャッシュ・env はすべて割当領域 `/mnt/gpu/workspace/2025/yuto_wada` 配下に置く。

```bash
# --- (Mac) 割当領域へ送信。編集後の再送も同じコマンド（差分のみ転送） ---
rsync -av --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
  /Users/wadaketsunin/research <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/

# --- (サーバー) セットアップ ---
ssh <user>@<host>
cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS
export HF_HOME=/mnt/gpu/workspace/2025/yuto_wada/hf_cache   # 重み等の DL 先を割当領域に隔離
export WANDB_MODE=disabled

# conda env（初回のみ。割当領域に作成）
conda env create -f environment.yml --prefix /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto
conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto

# --- (サーバー) 実行 ---
nvidia-smi                          # 空き GPU を確認
export CUDA_VISIBLE_DEVICES=0       # 空いている番号を指定
./run_sft_finetune.sh               # 学習 → models/ に保存
./run_sft_inference.sh              # 学習済みモデルを自動検出 → 推論 → 評価
cat output/sft_results/all_results.json  # FKGL 要求値 vs 達成値の相関/MAE を確認

# --- (Mac) 結果を回収 ---
rsync -av <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output/ \
  /Users/wadaketsunin/research/taming-CATS/output/
```

## ドキュメント

- [docs/direction.md](docs/direction.md) — 研究方針・提案手法・ロードマップ・実行フロー
- [docs/README.md](docs/README.md) — taming-CATS コードベース解説の索引
- [docs/experiment_flow.md](docs/experiment_flow.md) — データ分割→評価の全体像（mermaid）
- [docs/directory_roles.md](docs/directory_roles.md) — 各ディレクトリ・スクリプトの役割
- [docs/reproduction_guide.md](docs/reproduction_guide.md) — 再現手順と書き換え箇所
