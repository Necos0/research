# 数値情報を保持する制御トークンによるテキスト平易化

LLM によるテキスト平易化において、可読性レベルの制御に加えて **「原文中の数値情報を欠落させない」** ことを制御トークンで実現する研究のリポジトリ。先行研究 **Taming-CATS** の制御トークン機構を土台にする。

研究の狙い・提案手法・ロードマップ・実験の実行フロー（サーバー運用、git clone/pull、tmux、結果回収など）は **[docs/direction.md](docs/direction.md)** を参照。

## リポジトリ構成

| パス | 内容 |
| --- | --- |
| `taming-CATS/` | 実装本体（Taming-CATS 再現コード＋本研究の実験設定）。使用データは Med-EASi のみに整理済み |
| `docs/direction.md` | 研究方針・提案手法・ロードマップ・実験の実行フロー（メインの索引） |
| `docs/results_fkgl_medeasi_smoketest.html` | スモークテストの結果レポート |
| `results/` | サーバーから scp で回収した実験別の結果（`.gitignore` 済み） |
| `2604.01779v1.pdf` | 参照論文 |

## 現在の実験：Med-EASi × `<FKGL>` 再現（1B モデルでの実走行）

- **ブランチ**: `exp/fkgl-medeasi-1b`
- **目的**: ロードマップ1.2。スモークテストで疎通確認したパイプラインを、実運用サイズの 1B モデル・Med-EASi 全件で実走行し、`<FKGL>` 制御による再現結果を得る。
- **設定**:
  - モデル `meta-llama/Llama-3.2-1B-Instruct` / データ `medeasi`（ローカル）/ 制御属性 `FKGL`
  - 学習: train 667件・val 87件（全件）/ batch_size 4（gradient_accumulation 4 → 実効16）/ learning_rate 5e-6 / 3エポック / max_length 512
  - 推論: test 98件（全件）/ 1シード（seed=37）/ batch_size 16 / max_length 1024

### 特別な操作

標準の実行手順は [docs/direction.md](docs/direction.md)。この実験に固有の操作は次のとおり：

- **`meta-llama/Llama-3.2-1B-Instruct` は gated モデル**。事前に HF 上でライセンスを承認し、サーバー側で `export HF_TOKEN=<token>`（読み取り可トークン）を設定してから学習を回す。
- **推論時は BERTScore をスキップする**。評価の BERTScore が大語彙で GPU OOM を起こすため、環境変数 `SKIP_BERTSCORE=1` を付けて回す（`src/classes/Metrics.py` が参照）。
  ```bash
  SKIP_BERTSCORE=1 ./run_sft_inference.sh
  ```
