# 数値情報を保持する制御トークンによるテキスト平易化

LLM によるテキスト平易化において、可読性レベルの制御に加えて **「原文中の数値情報を欠落させない」** ことを制御トークンで実現する研究のリポジトリ。先行研究 **Taming-CATS** の制御トークン機構を土台にする。

研究の狙い・提案手法・実験の実行フロー（サーバー運用、git clone/pull、tmux、結果回収など）は **[docs/direction.md](docs/direction.md)**、中間発表までのロードマップは **[docs/roadmap.md](docs/roadmap.md)** を参照。

## リポジトリ構成

| パス | 内容 |
| --- | --- |
| `taming-CATS/` | 実装本体（Taming-CATS 再現コード＋本研究の実験設定）。使用データは Med-EASi のみに整理済み |
| `docs/direction.md` | 研究方針・提案手法・実験の実行フロー（メインの索引） |
| `docs/roadmap.md` | 中間発表までのロードマップ（進捗チェックリスト） |
| `docs/results_fkgl_medeasi_smoketest.html` | スモークテストの結果レポート |
| `results/` | サーバーから scp で回収した実験別の結果（`.gitignore` 済み） |
| `2604.01779v1.pdf` | 参照論文 |

## 現在の実験：Med-EASi × `<FKGL>` 1B フルデータ版・**プロンプト構築バグ修正後の再実行**

- **ブランチ**: `exp/fkgl-medeasi-1b-full-v2`
- **目的**: `exp/fkgl-medeasi-1b-full` の出力を全203件目視したところ、平易化が成立していなかった（予測の 202/203 が `The ` で始まる定型崩壊、原文コピー11件、事実の捏造・非文33件、数値保持 0.560）。原因は**制御トークンの配置バグとトークン化バグ**で、FKGL・KEEP の両ブランチが同じ共有コードを踏んでいた。本ブランチでそれを修正し、フルデータ版 FKGL を再学習・再推論して健全なベースラインを取り直す。**KEEP 側も同修正で再実行が必要**。両者の比較表（旧 `docs/comparison_fkgl_full_vs_keep.md`）は結論が支持できないため削除済みで、再実行後に作り直す。
- **修正内容**（3点。詳細は下記「修正したバグ」節）:
  1. `src/helpers/prompting.py` — 制御トークンを簡約文と同じ assistant ターンに置く（`continue_final_message=True`）
  2. `src/sft_finetune.py` / `src/sft_inference.py` — `add_special_tokens=False`（BOS の二重付与・completion への混入を防ぐ）
  3. `src/helpers/prompting.py` — 制御トークンと簡約文の間の空白を担保（トークン境界のズレ防止）
- **設定**（`METRIC_NAME=FKGL` 以外は `exp/keep-medeasi-1b` と完全に同一。ハイパラは v1 から変更なし＝**差分はバグ修正のみ**）:
  - モデル `meta-llama/Llama-3.2-1B-Instruct`（gated・HF トークン要）/ データ `medeasi`（ローカル `data/splits_flattened_full`＝フル未フィルタ）/ 制御属性 `FKGL`
  - 学習: train 1499件・val 191件（全件）/ batch_size 4（gradient_accumulation 4 → 実効16）/ learning_rate 5e-6 / 3エポック / max_length 512
  - 推論: test 203件（全件）/ 1シード（seed=37）/ batch_size 8 / max_length 1024（batch_size は 32GB GPU の OOM 対策。全プロンプトを max_length に固定パディングしているため greedy の生成結果はバッチサイズに依存しない）
- **評価での比較軸**: FKGL の制御精度（MAE）に加え、`prediction_metrics.keep`（数値保持率）が全事例で自動計算されるため、KEEP 1B の結果（`results/keep-medeasi-1b/`）と数値あり事例の保持率・SARI/LENS を同一 test 203件で直接比較できる。
- **再実行後に必ず確認すること**: SARI/LENS が論文値と一致していても品質は保証されない（旧実行では捏造事例が SARI 67.6 を記録していた）。**出力を必ず目視し**、`The ` 始まりの比率が参照文並み（約1割）に戻っているかを崩壊の指標として確認する。

### 修正したバグ（v1 = `exp/fkgl-medeasi-1b-full` の失敗原因）

いずれも FKGL/KEEP 共通の共有コードにあり、両実験が同じ壊れ方をしていた。

**1. 制御トークンが簡約文と別ターンに分離していた（最重要）**

`format_prompt_with_tokenizer` が制御トークンを「完結した assistant メッセージ」として渡し、かつ `add_generation_prompt=True` としていたため、chat template が `<|eot_id|>` でそのターンを閉じ、assistant ヘッダをもう一度開いていた。実際に送られていたプロンプト末尾:

```
<|start_header_id|>assistant<|end_header_id|>\n\n<FKGL=9.6><|eot_id|>   ← 制御トークンだけのターン（閉じている）
<|start_header_id|>assistant<|end_header_id|>\n\n                       ← ここから生成
```

モデルから見ると「`<FKGL=9.6>` とだけ言い終えた。これから新しい発言をゼロから始める」状態で、制御トークンは生成の接頭辞になっていない。制御トークンと最初の学習対象トークンの間に6トークンほどの機械的な記号が挟まり、条件付けがほぼ効かない。修正後:

```
<|start_header_id|>assistant<|end_header_id|>\n\n<FKGL=9.6> The risk of ...<|eot_id|>
                                                └制御トークン┘└─ 採点対象 ─┘
```

`continue_final_message=True` と `add_generation_prompt` は**排他**（両方 True だとエラー）なので後者を削除する。

**2. `add_special_tokens=True` で BOS が二重付与・completion に混入していた**

`prompt` は `apply_chat_template` が既に `<|begin_of_text|>` を含むのに再付与されて BOS が2つ並び、さらに `completion`（系列の途中）にも BOS が付いていた。`labels = [-100]*len(prompt_ids) + completion_ids` なので、**モデルは「答えの第一声は `<|begin_of_text|>`」と学習していた**。答えの書き出しを司る位置が汚染され、1 と併せて `The ` 100% の定型崩壊を招いた。

**3. 制御トークン末尾の空白（サイレントな罠）**

`control_token` は `f"<{metric}={value}> "` と末尾に空白を持つが、chat template は assistant の content を `| trim` する。この状態で `continue_final_message=True` にすると、transformers は「最終メッセージ content の**最後の出現位置**」で文字列を切り詰めるため、レンダリング済み文字列に `<FKGL=9.6> `（空白付き）が見つからず、**代わりに user メッセージの EXPLANATION 内にある同じ文字列にマッチして、そこでプロンプトを切断する**（assistant ターンごと消える）。例外は出ない。よって末尾の空白は除去必須で、代わりに `format_completion_with_tokenizer` で簡約文の先頭に空白を補う。空白の有無でトークン ID は変わる（`"By"`=1383 / `" By"`=3296）ため繋ぎ目は正確に合わせる。

**未修正（意図的）**: JSONL 生成時と実行時で textstat のバージョンが異なり、プロンプトに書く目標 FKGL（旧版・例 9.6）と MAE の正解値（新版・例 9.05）が別スケールになっている。ただし実測でズレは絶対平均 0.31・MAE への影響は 3.33→3.38 と小さく、ここを変えると KEEP 側の既存結果と指標が非互換になるため**今回は触らない**（`environment.yml` の textstat 未 pin は再現性上の課題として残る）。

### 参考：`<KEEP>` 制御トークンとフルデータ版スプリットの仕組み（keep系ブランチで追加済み・本実験ではデータのみ利用）

- **データ（2.0）**: `python src/add_keep_metric.py --dataset medeasi` で `source_metrics["keep"]` / `target_metrics["keep"]` に保持すべき数値の整形済み文字列（例 `"1995, 65"`、無ければ `"none"`）を追記する。**本ブランチのデータは付与済み**。タグと教師信号を整合させるため（FKGL タグが学習時に正解文の実際の値を使うのと同じ原則）、値はスプリットで使い分ける:
  - **train / val**: 原文の数値のうち**正解文が実際に保持している数値のみ**（Med-EASi の正解は数値あり事例の約4割で数値を落としており、原文の全数値をタグにすると教師信号が矛盾するため。1つも保持されていなければ `"none"` ＝保持制約なしの教師）
  - **test**: 原文の**全数値**（「全部保持せよ」という指示に従えるかを評価する）
- **フル（未フィルタ）データ版（2.0.1）**: `splits_flattened_filtered` は可読性フィルタで 852 件まで減っているが、`<KEEP>` 実験にこのフィルタは不要。`python src/build_keep_full_splits.py --dataset medeasi` で元データ全 1892 事例（1893 ペア）から keep 付きスプリット `data/splits_flattened_full/medeasi/`（train 1499 / val 191 / test 203）を生成する（**生成済み**）。スプリット割り当てはフィルタ済み版を引き継ぐ（旧 test ⊂ 新 test、train/test 間リークなし）ため、同一部分集合での比較評価が可能。加えて「数値なし／数値ありで正解が保持／数値ありで正解が全滅」の3グループの割合が全スプリットで全体比率に揃うよう層化してある（数値あり 22.8〜23.2%、正解保持＝train/val でタグが付く事例 16.6〜16.8%、全滅 6.2〜6.4%。FKGL 分布も各スプリットで均等）。学習・推論で使うデータの場所は、run スクリプト内の python 呼び出しに `--local_data_dir "data/splits_flattened_full"` としてベタ書きで指定する（`--control_tokens` 等と同じスタイル。省略時のデフォルトもフル版）。**本ブランチの run スクリプトはフル版を指定済み**。フィルタ済みに戻すには `data/splits_flattened_filtered` を指定する。学習と推論で同じ場所を指定すること。
- **プロンプト（2.0）**: `data/prompts/control_tokens.json` と `data/prompts/user_prompts.json` に `KEEP` を追加。入力の原文冒頭に `<KEEP=保持すべき数値>` を前置し、「これらの数値を必ず出力に保持せよ」と指示する（source ベース制御）。
- **数値の抽出・保持率**は `src/helpers/keep.py` に一元化（学習/推論データ付与と評価で同一ロジック）。評価では `Metrics.compute_metrics()` の `keep`（原文数値の保持率）を reference と prediction で比較する。

### 特別な操作

標準の実行手順は [docs/direction.md](docs/direction.md)。この実験に固有の操作は次のとおり：

- **`meta-llama/Llama-3.2-1B-Instruct` は gated モデル**。事前に HF 上でライセンスを承認し、サーバー側で `export HF_TOKEN=<token>`（読み取り可トークン）を設定してから学習を回す。
- フルデータ版スプリット（FKGL 値含む）はコミット済みなので、データの前処理は不要。
- **推論時の GPU メモリ対策はスクリプトに組み込み済み**（env の付け忘れで OOM した反省から `run_sft_inference.sh` 内で設定する）。素の `./run_sft_inference.sh` でよい。
  - `SKIP_BERTSCORE=1`: BERTScore は roberta-large を GPU にロードし OOM の主因になるためスキップ（backfill なし。この実験の比較指標では未使用）
  - `SKIP_LENS=1`（推論プロセスのみ）: LENS は推論中に計算せず、後段 `sft_eval.py` の backfill で LLM 解放後にまとめて計算する（LENS 値は summary に残る）
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` と `batch_size 8` で生成時のピークを抑制

### サーバー実行手順（この実験のコピペ用）

[docs/direction.md](docs/direction.md) の共通手順に、この実験の具体名（ブランチ `exp/fkgl-medeasi-1b-full-v2`）を当てはめたもの。上から順にコピペで実行できる。

**1.（Mac）コミットして GitHub へ push**

```bash
cd /Users/wadaketsunin/research
git add -A && git commit -m "exp: Med-EASi x FKGL 1B フルデータ版（プロンプト構築バグ修正後の再実行）"   # 未コミットの変更があれば
git push -u origin exp/fkgl-medeasi-1b-full-v2
```

**2.（サーバー）ブランチを引いて tmux ＋ conda 環境を準備**

```bash
ssh wada_yuto@calc40
```

```bash
cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS
git fetch origin && git switch exp/fkgl-medeasi-1b-full-v2
git pull                                  # 同一ブランチの更新を取り込む場合
rm -rf output models logs                 # 前実験の生成物を掃除（回収済みが前提。logs も消さないと前実験のログが混ざる）
tmux new -s exp-fkgl-medeasi-1b-full-v2
```

```bash
# ここから tmux セッション内
conda activate /mnt/gpu/workspace/2025/yuto_wada/envs/wada-yuto
export HF_TOKEN=<token>                   # gated モデル（Llama-3.2-1B）用。読み取り可トークン
```

**3.（サーバー・tmux 内）学習 → 推論を実行**

```bash
nvidia-smi                                # 空き GPU を確認
export CUDA_VISIBLE_DEVICES=0             # 空いている番号に書き換える
./run_sft_finetune.sh && ./run_sft_inference.sh   # 学習→推論を連続実行（学習が失敗したら推論には進まない）
```

推論のOOM対策（`SKIP_BERTSCORE` 等）は `run_sft_inference.sh` 内で設定済みなので、env プレフィックスは不要。

実行が始まったら `Ctrl-b` → `d` で detach して SSH を切ってよい。進捗確認は:

```bash
tmux attach -t exp-fkgl-medeasi-1b-full-v2
```

**4.（Mac）結果を回収して後片付け**

```bash
mkdir -p /Users/wadaketsunin/research/results/fkgl-medeasi-1b-full-v2
scp -r wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/models \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/logs \
  /Users/wadaketsunin/research/results/fkgl-medeasi-1b-full-v2/
```

```bash
# （サーバー）終わったセッションを削除
tmux kill-session -t exp-fkgl-medeasi-1b-full-v2
```

※ モデル重み（`*.safetensors`）が不要なら scp の `models` 行を外し、評価サマリ（`output/sft_results/all_results.json`）とログだけ回収してもよい。
