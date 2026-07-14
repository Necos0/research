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

## 現在の実験：Med-EASi × `<KEEP>` 1B フルデータ版・**プロンプト構築バグ修正後の再実行**（ロードマップ3.5）

- **ブランチ**: `exp/keep-medeasi-1b-v2`（`exp/fkgl-medeasi-1b-full-v2` から分岐）
- **目的**: 旧 KEEP 1B（`exp/keep-medeasi-1b`）は FKGL 側と同じ共有コードのプロンプト構築バグを踏んでおり、平易化が成立していなかった。修正済みコードで KEEP を再学習・再推論し、**FKGL v2 と直接比較できる健全な結果**を得る。比較表（3.6）はこの結果が出てから作り直す。
- **差分は `METRIC_NAME=KEEP` のみ**。分岐元の FKGL v2 とハイパラ・データ・修正コードがすべて同一なので、タグ以外の交絡はない。
- **設定**:
  - モデル `meta-llama/Llama-3.2-1B-Instruct`（gated・HF トークン要）/ データ `medeasi`（ローカル `data/splits_flattened_full`＝フル未フィルタ）/ 制御属性 `KEEP`
  - 学習: train 1499件・val 191件（全件）/ batch_size 4（gradient_accumulation 4 → 実効16）/ learning_rate 5e-6 / 3エポック / max_length 512
  - 推論: test 203件（全件）/ 1シード（seed=37）/ batch_size 8 / max_length 1024（batch_size は 32GB GPU の OOM 対策。全プロンプトを max_length に固定パディングしているため greedy の生成結果はバッチサイズに依存しない）
- **バグ修正が KEEP でも効くことは検証済み**: KEEP のタグ値は float ではなく文字列（`4` / `400, 20` / `none`）なので、修正3（`continue_final_message` の文字列マッチ）が壊れないかを実際の tokenizer でレンダリングして確認した。3パターンとも制御トークンが簡約文の接頭辞になり（assistant ヘッダは1つ・`<|eot_id|>` の割り込みなし）、completion 先頭への BOS 混入もない。
- **タグ値の設計（再掲）**: train/val は「正解文が実際に保持している数値のみ」、test は「原文の全数値」。データ側でこの通りになっていることも検証済み（train/val 100%一致）。
- **比較の際に必ず見ること**（FKGL v2 の解析で判明）: FKGL v2 は定型崩壊こそ直ったが、**予測の 54.2%（110/203）が原文と完全一致**しており、その副作用で数値保持率が 0.915（参照文 0.617）まで上がっている。**原文をコピーするほど保持率は自動的に上がる**ため、`<KEEP>` の効果は保持率だけでは測れない。比較表には必ず **原文コピー率**を併記し、両者のコピー率が同水準であることを確認したうえで保持率を比べること。
- **実行後に必ず確認すること**: SARI/LENS が論文値と一致していても品質は保証されない（旧実行では捏造事例が SARI 67.6 を記録していた）。**出力を必ず目視し**、`The ` 始まりの比率が参照文並み（約1割）に戻っているかを崩壊の指標として確認する。

### 直前の実験：Med-EASi × `<FKGL>` 1B フルデータ版 v2（ロードマップ3.4・完了）

`exp/fkgl-medeasi-1b-full-v2`。結果は `results/fkgl-medeasi-1b-full-v2/`。本 KEEP 実験の比較対象（ベースライン）。

- **定型崩壊は解消**: `The ` 始まりが 202/203（99.5%）→ **22/203（10.8%）** と参照文（10.3%）並みに回復。制御トークンの出力漏れ 0件。
- SARI **51.24** / LENS **56.79** / FKGL MAE **2.92**（v1 の SARI 40.66 より改善、論文水準）。BERTScore も backfill で復活（to_ref 0.905）。
- **残る懸念**: 原文コピー 54.2%（参照文は 0%）・空出力6件・FKGL は原文 14.11 → 予測 12.74 とアンダーシュート（参照文 11.88）。early stopping が epoch 1.38 で止まっており under-fit 気味の可能性。
- **これにより旧 3.2 の結論は覆る**: FKGL の数値保持率が 0.560 → **0.915** に上がったため、「KEEP 優位（0.560→0.674）」は成立しない。

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

### `<KEEP>` 制御トークンとフルデータ版スプリットの仕組み（本実験の中核）

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
  - `SKIP_BERTSCORE=1` / `SKIP_LENS=1`（**推論プロセスのみ**。`export` すると後段の評価まで止まるので前置きで渡す）: BERTScore（roberta-large）と LENS は評価モデルを GPU に載せるため、生成中の LLM と取り合って OOM の主因になる。推論中は計算せず、後段 `sft_eval.py` の backfill で **LLM 解放後にまとめてバッチ計算**する。値は `output_averaged.json` と summary（`output/sft_results/all_results.json`）に載る（`BERTScore`＝予測 vs 原文、`BERTScore_ref`＝予測 vs 正解。ブートストラップ信頼区間つき）。
    - backfill は `enrich_predictions_with_bertscore()` / `enrich_predictions_with_lens()`（`src/sft_eval.py`）。同一ペアはキャッシュで重複計算を避け、`batch_size=64` で `compute()` を呼ぶ（1件ずつ呼ぶと 203件×2指標＝406回になる）。
    - **注意**: 推論プロセスで計算をスキップするため、per-item の `output_*.json` には `BERTScore: 0.0` / `LENS: null` が残る。実値は `output_averaged.json` を見ること。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` と `batch_size 8` で生成時のピークを抑制

### サーバー実行手順（この実験のコピペ用）

[docs/direction.md](docs/direction.md) の共通手順に、この実験の具体名（ブランチ `exp/keep-medeasi-1b-v2`）を当てはめたもの。上から順にコピペで実行できる。

**1.（Mac）コミットして GitHub へ push**

```bash
cd /Users/wadaketsunin/research
git add -A && git commit -m "exp: Med-EASi x KEEP 1B フルデータ版（プロンプト構築バグ修正後の再実行）"   # 未コミットの変更があれば
git push -u origin exp/keep-medeasi-1b-v2
```

**2.（サーバー）ブランチを引いて tmux ＋ conda 環境を準備**

```bash
ssh wada_yuto@calc40
```

```bash
cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS
git fetch origin && git switch exp/keep-medeasi-1b-v2
git pull                                  # 同一ブランチの更新を取り込む場合
rm -rf output models logs                 # 前実験（FKGL v2）の生成物を掃除。**必須**：
                                          #   消さないと run_sft_inference.sh の MODEL_DIR 自動選択が
                                          #   FKGL のモデルを拾い、logs も混ざる（回収済みが前提）
tmux new -s exp-keep-medeasi-1b-v2
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
tmux attach -t exp-keep-medeasi-1b-v2
```

**4.（Mac）結果を回収して後片付け**

```bash
mkdir -p /Users/wadaketsunin/research/results/keep-medeasi-1b-v2
scp -r wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/models \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/logs \
  /Users/wadaketsunin/research/results/keep-medeasi-1b-v2/
```

```bash
# （サーバー）終わったセッションを削除
tmux kill-session -t exp-keep-medeasi-1b-v2
```

※ モデル重み（`*.safetensors`）が不要なら scp の `models` 行を外し、評価サマリ（`output/sft_results/all_results.json`）とログだけ回収してもよい。
