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
- サーバーの割当領域 **`/mnt/gpu/workspace/2025/yuto_wada`** 配下に、リポジトリ・HF キャッシュ・conda env を**すべて置く**（共有ストレージを圧迫しない）。
- コードの受け渡しは **GitHub 経由の `git clone` / `git pull`**。Mac 側で `exp/...` ブランチを編集・コミットして push、サーバーで取り込む。公開リポジトリなので **HTTPS 読み取りのみ**（認証情報を共有マシンに置かない）。結果（モデル重み等、git 非管理）は **scp** で Mac 側の `results/<実験名>/`（`.gitignore` 済みの実験別アーカイブ）へ回収する。
- **各ブランチ＝各実験**。サーバーの clone は **1個だけ**置き、実験ごとに `git fetch origin && git switch exp/...` で対象ブランチを引いて切り替える。`output/`・`models/` はブランチ切り替えで消えないため、**回収済みなら削除してから**次の実験を回す（推論が別実験のモデルを拾うのを防ぐ）。

```
（Mac）編集・コミット・push → GitHub
   →（サーバー）git clone/pull → conda env 構築 → 実験実行 → scp で結果を回収
```

conda 仮想環境は `taming-CATS/environment.yml`（conda-forge ベースで `cuda-toolkit`＋`pytorch` を同梱）から構築する。詳しい方針は [docs/direction.md](docs/direction.md)。

## 現在の実験：Med-EASi × `<FKGL>` 再現（最小GPUスモークテスト）

- **ブランチ**: `exp/fkgl-medeasi-repro`
- **目的**: データ準備→学習→推論→評価のパイプラインが端から端まで通ることを確認する（数値の再現ではなく疎通確認）。
- **設定**: モデル `Qwen/Qwen3-0.6B` / データ `medeasi`（ローカル）/ 制御属性 `FKGL` / train 16件・val 8件・1エポック・max_length 1024 / 推論は1シード・test 8件。

### 研究室サーバーでの実行

`<user>@<host>` は研究室サーバーに置き換える。リポジトリ・キャッシュ・env はすべて割当領域 `/mnt/gpu/workspace/2025/yuto_wada` 配下に置く。

```bash
# --- (Mac) コミットして GitHub へ push ---
git push -u origin exp/fkgl-medeasi-repro

# --- (サーバー) 取得 ---
ssh <user>@<host>
cd /mnt/gpu/workspace/2025/yuto_wada
git clone https://github.com/Necos0/research.git        # 初回のみ（公開リポジトリ・認証不要）
cd research/taming-CATS
git fetch origin && git switch exp/fkgl-medeasi-repro    # 対象ブランチに切り替え（同一ブランチ更新時は git pull）
# 別実験から切り替えた場合は前実験の生成物を掃除（回収済みが前提）: rm -rf output models

# --- (サーバー) conda env（初回のみ。割当領域に作成し、環境変数を env に登録） ---
export HF_HOME=/mnt/gpu/workspace/2025/yuto_wada/hf_cache   # env 作成時の DL を割当領域へ
conda env create -f environment.yml --prefix /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto
conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto
conda env config vars set HF_HOME=/mnt/gpu/workspace/2025/yuto_wada/hf_cache WANDB_MODE=disabled
conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto   # 反映のため再 activate

# --- (サーバー) 実行（長時間ジョブは tmux 内で回す） ---
tmux new -s exp-fkgl-medeasi-repro  # 再接続: tmux attach -t exp-fkgl-medeasi-repro
nvidia-smi                          # 空き GPU を確認
export CUDA_VISIBLE_DEVICES=0       # 空いている番号を指定
./run_sft_finetune.sh               # 学習 → models/ に保存
./run_sft_inference.sh              # 学習済みモデルを自動検出 → 推論 → 評価
cat output/sft_results/all_results.json  # FKGL 要求値 vs 達成値の相関/MAE を確認
# 実行を開始したら Ctrl-b d で detach → SSH を切ってよい

# --- (Mac) 結果を回収（実験別に results/<実験名>/ へまとめる。ローカルに output/models は置かない） ---
mkdir -p /Users/wadaketsunin/research/results/fkgl-medeasi-repro
scp -r <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output \
  <user>@<host>:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/models \
  /Users/wadaketsunin/research/results/fkgl-medeasi-repro/
```

## ドキュメント

- [docs/direction.md](docs/direction.md) — 研究方針・提案手法・ロードマップ・実行フロー
- [docs/README.md](docs/README.md) — taming-CATS コードベース解説の索引
- [docs/experiment_flow.md](docs/experiment_flow.md) — データ分割→評価の全体像（mermaid）
- [docs/directory_roles.md](docs/directory_roles.md) — 各ディレクトリ・スクリプトの役割
- [docs/reproduction_guide.md](docs/reproduction_guide.md) — 再現手順と書き換え箇所
