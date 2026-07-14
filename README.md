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
| `results/` | サーバーから rsync で回収した実験別の結果（`.gitignore` 済み。モデル重みは除外する） |
| `2604.01779v1.pdf` | 参照論文 |

## 現在の実験：Med-EASi × `<KEEP>` 1B フルデータ版・**再実行（キャッシュ事故のやり直し）**（ロードマップ3.5）

> **⚠️ `results/keep-medeasi-1b-v2/` の結果は無効。** datasets キャッシュ事故（下記「修正したバグ 4」）により、**修正前（7/9）のプロンプトで学習していた**。症状は **62.1%（126/203）が空出力**（`<KEEP=none>` では 76.9%）、SARI 35.5・BERTScore 0.375 と全面崩壊。**比較対象の `results/fkgl-medeasi-1b-full-v2/` も同じ事故で無効。両方とも再実行が必要。**

- **ブランチ**: `exp/keep-medeasi-1b-v2`（`exp/fkgl-medeasi-1b-full-v2` から分岐）
- **目的**: 修正済みコードで KEEP を再学習・再推論し、**FKGL v2 と直接比較できる健全な結果**を得る。比較表（3.6）はこの結果が出てから作る。
- **差分は `METRIC_NAME=KEEP` のみ**。分岐元の FKGL v2 とハイパラ・データ・修正コードがすべて同一なので、タグ以外の交絡はない。
- **設定**:
  - モデル `meta-llama/Llama-3.2-1B-Instruct`（gated。`HF_TOKEN` は conda env が設定するので手動 export は不要）/ データ `medeasi`（ローカル `data/splits_flattened_full`＝フル未フィルタ）/ 制御属性 `KEEP`
  - 学習: train 1499件・val 191件（全件）/ batch_size 4（gradient_accumulation 4 → 実効16）/ learning_rate 5e-6 / 3エポック / max_length 512
  - 推論: test 203件（全件）/ 1シード（seed=37）/ batch_size 8 / max_length 1024（batch_size は 32GB GPU の OOM 対策。全プロンプトを max_length に固定パディングしているため greedy の生成結果はバッチサイズに依存しない）
- **バグ修正が KEEP でも効くことは検証済み**: KEEP のタグ値は float ではなく文字列（`4` / `400, 20` / `none`）なので、修正3（`continue_final_message` の文字列マッチ）が壊れないかを実際の tokenizer でレンダリングして確認した。3パターンとも制御トークンが簡約文の接頭辞になり（assistant ヘッダは1つ・`<|eot_id|>` の割り込みなし）、completion 先頭への BOS 混入もない。
- **タグ値の設計（再掲）**: train/val は「正解文が実際に保持している数値のみ」、test は「原文の全数値」。データ側でこの通りになっていることも検証済み（train/val 100%一致）。
- **実行後に必ず確認すること**:
  1. 学習開始直後に **`--- prompt sanity OK: ...'<KEEP=none>'`** が出ること（出なければプロンプト構築が壊れている）。
  2. 学習ログの DEBUG に出るプロンプトの **`Today Date` が実行日と一致**すること（古ければキャッシュを踏んでいる）。
  3. 出力を**目視**し、**空出力がほぼ0**・`The ` 始まりが参照文並み（約1割）・原文コピーが過大でないこと。SARI/LENS が論文値と一致していても品質は保証されない。
- **評価時の注意（分析で判明）**:
  - **数値保持率は単独では `<KEEP>` の効果を示さない**。原文をコピーするほど保持率は自動的に上がるため、必ず **原文コピー率・空出力数とセット**で報告する。
  - 数値あり47件のうち、**識別的なのは22件だけ**（参照文が数値を落とす事例）。残り25件は参照文も全保持で誰でも 1.000 になり、混ぜると差が薄まる。**22件を主軸に見る**。
  - 保持率は「マクロ平均（現行）」「**ミクロ＝何個中何個**」「**完全保持事例率**」を出す。現行コードはマクロのみ。
  - **`losses.MAE` は KEEP では壊れている**: 参照文を目標として扱うため、保持率100%を達成すると MAE が 0.383 と**悪化**する（成功を罰する指標）。KEEP では使わない。

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

**4. datasets の `map` キャッシュが 1・3 の修正を無効化していた（v2 が壊れた原因）**

`sft_finetune.py` の `dataset.map(process_instance)` は**マップ関数のフィンガープリント**でキャッシュする。プロンプトを実際に組み立てているのは `helpers/prompting.py` の関数だが、`map` に渡す `process_instance` 自身のバイトコードは変わらないため、**`prompting.py` を直してもフィンガープリントが変わらず、過去実行のキャッシュがヒットする**。

その結果、FKGL v2 / KEEP v2 はどちらも **7/9（修正前）のプロンプトで学習**していた。学習ログの DEBUG に証拠が残る:

```
Today Date: 09 Jul 2026        ← 実行日は 7/13, 7/14
<KEEP=none><|eot_id|><|start_header_id|>assistant<|end_header_id|>   ← 修正1 が効いていない
```

一方 `tokenize` は関数本体を書き換えた（`add_special_tokens=False`）のでフィンガープリントが変わり再実行された。よって**修正2 だけが効き、修正1・3 は効かない**という中途半端な状態になった。推論側はプロンプトを組み直すため、**学習と推論でプロンプト構造が食い違う**。

- KEEP v2: **62.1%（126/203）が空出力**。モデルは「制御トークンの直後は `<|eot_id|>`」と学習しており、推論で「続きを書け」と言われて即 EOS を吐いた（`<KEEP=none>` では 76.9%）。
- FKGL v2: 原文コピー 54.2%。`The ` 崩壊が直って見えたのも同じ食い違いの産物で、健全ではない。

**この事故は `eval_loss` でも SARI でも検知できなかった**（壊れたデータで学習し、同じ壊れたデータで検証していたため辻褄が合っていた）。対策として全 `.map()` に `load_from_cache_file=False` を付け、さらに**組み立てたプロンプトが「assistant ターン1つ」かつ「制御トークンで終わる」ことを assert で検証**するガードを学習・推論の両方に入れた。`run_experiment.sh` は HF datasets キャッシュも削除する。

**教訓**: バグを直したあとは、**モデルが実際に食べたプロンプト（学習ログの DEBUG）を必ず目視する**。指標だけ見ても検知できない。

**未修正（意図的）**: JSONL 生成時と実行時で textstat のバージョンが異なり、プロンプトに書く目標 FKGL（旧版・例 9.6）と MAE の正解値（新版・例 9.05）が別スケールになっている。ただし実測でズレは絶対平均 0.31・MAE への影響は 3.33→3.38 と小さく、ここを変えると既存結果と指標が非互換になるため**今回は触らない**（`environment.yml` の textstat 未 pin は再現性上の課題として残る）。

### `<KEEP>` 制御トークンとフルデータ版スプリットの仕組み（本実験の中核）

- **データ（2.0）**: `python src/add_keep_metric.py --dataset medeasi` で `source_metrics["keep"]` / `target_metrics["keep"]` に保持すべき数値の整形済み文字列（例 `"1995, 65"`、無ければ `"none"`）を追記する。**本ブランチのデータは付与済み**。タグと教師信号を整合させるため（FKGL タグが学習時に正解文の実際の値を使うのと同じ原則）、値はスプリットで使い分ける:
  - **train / val**: 原文の数値のうち**正解文が実際に保持している数値のみ**（Med-EASi の正解は数値あり事例の約4割で数値を落としており、原文の全数値をタグにすると教師信号が矛盾するため。1つも保持されていなければ `"none"` ＝保持制約なしの教師）
  - **test**: 原文の**全数値**（「全部保持せよ」という指示に従えるかを評価する）
- **フル（未フィルタ）データ版（2.0.1）**: `splits_flattened_filtered` は可読性フィルタで 852 件まで減っているが、`<KEEP>` 実験にこのフィルタは不要。`python src/build_keep_full_splits.py --dataset medeasi` で元データ全 1892 事例（1893 ペア）から keep 付きスプリット `data/splits_flattened_full/medeasi/`（train 1499 / val 191 / test 203）を生成する（**生成済み**）。スプリット割り当てはフィルタ済み版を引き継ぐ（旧 test ⊂ 新 test、train/test 間リークなし）ため、同一部分集合での比較評価が可能。加えて「数値なし／数値ありで正解が保持／数値ありで正解が全滅」の3グループの割合が全スプリットで全体比率に揃うよう層化してある（数値あり 22.8〜23.2%、正解保持＝train/val でタグが付く事例 16.6〜16.8%、全滅 6.2〜6.4%。FKGL 分布も各スプリットで均等）。学習・推論で使うデータの場所は、run スクリプト内の python 呼び出しに `--local_data_dir "data/splits_flattened_full"` としてベタ書きで指定する（`--control_tokens` 等と同じスタイル。省略時のデフォルトもフル版）。**本ブランチの run スクリプトはフル版を指定済み**。フィルタ済みに戻すには `data/splits_flattened_filtered` を指定する。学習と推論で同じ場所を指定すること。
- **プロンプト（2.0）**: `data/prompts/control_tokens.json` と `data/prompts/user_prompts.json` に `KEEP` を追加。入力の原文冒頭に `<KEEP=保持すべき数値>` を前置し、「これらの数値を必ず出力に保持せよ」と指示する（source ベース制御）。
- **数値の抽出・保持率**は `src/helpers/keep.py` に一元化（学習/推論データ付与と評価で同一ロジック）。評価では `Metrics.compute_metrics()` の `keep`（原文数値の保持率）を reference と prediction で比較する。

### 特別な操作

標準の実行手順は [docs/direction.md](docs/direction.md)。この実験に固有の操作は次のとおり：

- **`meta-llama/Llama-3.2-1B-Instruct` は gated モデル**だが、`HF_TOKEN` は **conda env の activate 時に設定される**ので手動の `export` は不要。
- **`CUDA_VISIBLE_DEVICES` の指定も不要**（GPU はサーバー側で割り当てられる）。
- フルデータ版スプリット（FKGL 値含む）はコミット済みなので、データの前処理は不要。
- **推論時の GPU メモリ対策はスクリプトに組み込み済み**（env の付け忘れで OOM した反省から `run_sft_inference.sh` 内で設定する）。素の `./run_sft_inference.sh` でよい。
  - `SKIP_BERTSCORE=1` / `SKIP_LENS=1`（**推論プロセスのみ**。`export` すると後段の評価まで止まるので前置きで渡す）: BERTScore（roberta-large）と LENS は評価モデルを GPU に載せるため、生成中の LLM と取り合って OOM の主因になる。推論中は計算せず、後段 `sft_eval.py` の backfill で **LLM 解放後にまとめてバッチ計算**する。値は `output_averaged.json` と summary（`output/sft_results/all_results.json`）に載る（`BERTScore`＝予測 vs 原文、`BERTScore_ref`＝予測 vs 正解。ブートストラップ信頼区間つき）。
    - backfill は `enrich_predictions_with_bertscore()` / `enrich_predictions_with_lens()`（`src/sft_eval.py`）。同一ペアはキャッシュで重複計算を避け、`batch_size=64` で `compute()` を呼ぶ（1件ずつ呼ぶと 203件×2指標＝406回になる）。
    - **注意**: 推論プロセスで計算をスキップするため、per-item の `output_*.json` には `BERTScore: 0.0` / `LENS: null` が残る。実値は `output_averaged.json` を見ること。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` と `batch_size 8` で生成時のピークを抑制

### サーバー実行手順（この実験のコピペ用）

**サーバー側は `run_experiment.sh` の1コマンドで完結する。** 手で `export` するものは無い（`HF_TOKEN` / `HF_HOME` / `WANDB_MODE` は conda env の activate 時に設定済み。GPU もサーバーが割り当てるので `CUDA_VISIBLE_DEVICES` は不要）。

**1.（Mac）コミットして GitHub へ push**

```bash
cd /Users/wadaketsunin/research
git add -A && git commit -m "exp: <条件の説明>"   # 未コミットの変更があれば
git push -u origin exp/keep-medeasi-1b-v2
```

**2.（サーバー）1コマンドで実行**

```bash
ssh wada_yuto@calc40
cd /mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS \
  && git fetch origin && git switch exp/keep-medeasi-1b-v2 && git merge --ff-only origin/exp/keep-medeasi-1b-v2 \
  && ./run_experiment.sh exp/keep-medeasi-1b-v2
```

先頭の `git fetch/switch/merge` は必ず付ける。サーバーのローカルブランチが古いと `run_experiment.sh` 自体がまだ無く `No such file or directory` になるため（スクリプトはブランチを切り替える側なので、自分自身を先に持ってこられない）。

`git pull` ではなく **`git merge --ff-only origin/<branch>`** を使うこと。`git pull --ff-only` は upstream（追跡情報）が未設定のローカルブランチだと `exit 1` で落ちる。`run_experiment.sh` の中でも同じ理由で `git merge --ff-only` にしてある（`set -e` で走るため、ここで落ちるとスクリプトごと死ぬ）。

これだけで次を全部やる（特定の GPU に固定したいときだけ第2引数で指定: `./run_experiment.sh exp/keep-medeasi-1b-v2 1`）:

1. ブランチを `fetch` / `switch` / `pull` し、**そのブランチ版の本スクリプトで実行し直す**
2. **tmux セッションを自動で張る**（SSH が切れても継続。`Ctrl-b` → `d` で detach）
3. conda env を有効化（`HF_TOKEN` 等は env が設定するので手動 export は不要）
4. 前実験の `output/` `models/` `logs/` と **HF datasets キャッシュ**を掃除（削除前に確認を求める）
5. **学習 → 推論**を連続実行（学習が失敗したら推論に進まない）
6. Mac へ結果を回収する rsync コマンドを表示（モデル重みは除外）

**datasets キャッシュの削除は必須**。残すと `map` のフィンガープリントが変わらない場合に**修正前のプロンプトで学習してしまう**（FKGL v2 / KEEP v2 はこれで壊れた。詳細は下記「修正したバグ」節）。コード側でも `load_from_cache_file=False` にしてあるが、スクリプトでも二重に潰している。

**学習開始直後に `--- prompt sanity OK: ...'<KEEP=none>'` が出ることを必ず確認する。** 出ない／assert で落ちる場合は、プロンプト構築が壊れている。

進捗確認・再接続:

```bash
tmux attach -t exp-keep-medeasi-1b-v2      # セッション名は exp- + ブランチ名（exp/ を除く）
```

**3.（Mac）結果を回収（モデル重みは除外する）**

完了時にスクリプトが表示するコマンドをそのまま貼ればよい。

```bash
mkdir -p /Users/wadaketsunin/research/results/keep-medeasi-1b-v2
rsync -av --exclude='*.safetensors' \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/output \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/models \
  wada_yuto@calc40:/mnt/gpu/workspace/2025/yuto_wada/research/taming-CATS/logs \
  /Users/wadaketsunin/research/results/keep-medeasi-1b-v2/
```

**モデル重み（`*.safetensors`）は回収しない。** 1実験あたり約4.9GB とアーカイブの99%以上を占めるが、**Mac には CUDA が無く使い道がない**。再現性は「ブランチ＝実験」で担保される。`models/` の `args.json` / `config.json` / `tokenizer.json` は残すので、実験条件の証跡とローカルでのプロンプト再現には使える（除外により 4.9GB → 約20MB）。

**例外**: 学習し直さずに**推論だけやり直す**可能性がある場合は、その実験の `models/` を**サーバーに残したままにする**（次の実験を回すと `run_experiment.sh` が消す）。重みを Mac に持ってきても、結局サーバーに戻さないと推論できない。
