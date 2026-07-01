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

## 現在の実験：Med-EASi × `<FKGL>` 再現（最小GPUスモークテスト）

- **ブランチ**: `exp/fkgl-medeasi-repro`
- **目的**: データ準備→学習→推論→評価のパイプラインが端から端まで通ることを確認する（数値の再現ではなく疎通確認）。
- **設定**: モデル `Qwen/Qwen2.5-0.5B-Instruct` / データ `medeasi`（ローカル）/ 制御属性 `FKGL` / train 16件・val 8件・1エポック・max_length 512 / 推論は1シード（seed=37）・test 8件。

### 特別な操作

標準の実行手順は [docs/direction.md](docs/direction.md)。この実験に固有の操作は次のみ：

- **推論時は BERTScore をスキップする**。評価の BERTScore が語彙15万規模で GPU OOM を起こすため、環境変数 `SKIP_BERTSCORE=1` を付けて回す（`src/classes/Metrics.py` が参照）。
  ```bash
  SKIP_BERTSCORE=1 ./run_sft_inference.sh
  ```
