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

- 学習・推論・評価は **研究室サーバー（CUDA GPU）** で実行。手元の Mac は編集と git 操作のみ。
- **実験ごとにブランチを切る**（`main` は常に動く状態に保つ）。

```
（Mac）ブランチ作成・編集・push  →  （サーバー）SSH → 該当ブランチを pull
     → conda env 構築 → 実験実行 → 結果サマリを push で回収
```

各自サーバー上で `taming-CATS/environment.yml` から conda 仮想環境を構築する
（conda-forge ベースで `cuda-toolkit`＋`pytorch` を同梱）。詳しい手順は [docs/direction.md](docs/direction.md#実験の実行フロー共通運用)。

## 現在の実験：Med-EASi × `<FKGL>` 再現（最小GPUスモークテスト）

- **ブランチ**: `exp/fkgl-medeasi-repro`
- **目的**: データ準備→学習→推論→評価のパイプラインが端から端まで通ることを確認する（数値の再現ではなく疎通確認）。
- **設定**: モデル `Qwen/Qwen3-0.6B` / データ `medeasi`（ローカル）/ 制御属性 `FKGL` / train 16件・val 8件・1エポック・max_length 1024 / 推論は1シード・test 8件。

### 研究室サーバーでの実行

```bash
git clone git@github.com:Necos0/research.git
cd research/taming-CATS
git switch exp/fkgl-medeasi-repro

conda env create -f environment.yml      # 初回のみ
conda activate wada-yuto

./run_sft_finetune.sh                    # 学習 → models/ に保存
./run_sft_inference.sh                   # 学習済みモデルを自動検出 → 推論 → 評価
cat output/sft_results/all_results.json  # FKGL 要求値 vs 達成値の相関/MAE を確認
```

## ドキュメント

- [docs/direction.md](docs/direction.md) — 研究方針・提案手法・ロードマップ・実行フロー
- [docs/README.md](docs/README.md) — taming-CATS コードベース解説の索引
- [docs/experiment_flow.md](docs/experiment_flow.md) — データ分割→評価の全体像（mermaid）
- [docs/directory_roles.md](docs/directory_roles.md) — 各ディレクトリ・スクリプトの役割
- [docs/reproduction_guide.md](docs/reproduction_guide.md) — 再現手順と書き換え箇所
