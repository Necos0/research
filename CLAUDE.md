# CLAUDE.md

このリポジトリで作業を始める前に、必ず次の2つを読み込んでから着手すること。

1. [`docs/direction.md`](docs/direction.md) — 研究の背景・目的・課題・提案手法・実験の実行フロー（共通運用）
2. [`docs/roadmap.md`](docs/roadmap.md) — 次の中間発表に向けたロードマップ（進捗チェックリスト）。1回目の中間発表は終了済み

これらが研究の方向性と現在地を定義する。タスクに取りかかる際は、まず両ファイルを参照して文脈を把握すること。

## 共通文書は main が正本（worktree は使わない）

`CLAUDE.md`・`docs/direction.md`・`docs/roadmap.md` は全実験に共通の文書なので、**main ブランチだけが正本**。
**worktree は使わない**（ユーザーの指示）。チェックアウトは `/Users/wadaketsunin/research` の1か所だけで、ブランチを切り替えて使う。

- **読むとき**: main にいればそのまま `docs/` を読む。`exp/*` にいるときは切り替えずに `git show main:docs/direction.md`・`git show main:docs/roadmap.md` で main 版を読む。
- **書くとき**: `git switch main` してから編集し、**main にコミット**する（push はユーザーの了承を得てから）。終わったら元の `exp/*` に戻る。未コミットの変更があって切り替えられないときは、勝手に stash や破棄をせずユーザーに確認する。`exp/*` ブランチ側の共通文書は編集しない。
- **`exp/*` ブランチにある共通文書は、main を取り込んだ時点の写し**で、古いことがある。読まない・編集しない。
- **実験ブランチへは `git merge main` で取り込む**。main のコードは古いので、実験ブランチは従来どおり元の実験ブランチから切る。切ったら `git merge main` で共通文書とこの CLAUDE.md を入れる。main には共通文書のコミットしか積まないので、このマージで変わるのは共通文書だけ。衝突したら main 側を採用する（`git checkout --theirs CLAUDE.md docs/direction.md docs/roadmap.md`）。
- 実験固有の文書（README の「現在の実験」節、`docs/results_*.html` など）は、従来どおり各 `exp/*` ブランチに置く。
- `results/`（git 管理外）はブランチを切り替えても残るので、main にいても全実験の結果を読める。

## 卒論も main が正本（実験が終わっても main に残す）

卒論（LaTeX）と卒論メモは全実験に共通なので、共通文書と同じく **main だけが正本**。卒論を書くときは `git switch main` して `docs/thesis/` を編集し、main にコミットする。`git merge main` で `exp/*` に入った写しは読まない・編集しない。

- `docs/thesis/` — 卒論本体。研究室テンプレート（inouelab-waseda/tex-template）由来、upLaTeX → biber → dvipdfmx。VS Code で `/Users/wadaketsunin/research` を開いたまま `docs/thesis/sample.tex` を保存すると自動ビルドされる（設定は `docs/thesis/.latexmkrc` と、git 管理外のローカル `.vscode/settings.json`）
- `docs/卒論.md`（先生の指示）、`docs/卒論_構成案.md`（章立て）
- `docs/卒論_先輩/` — 先輩の卒論 PDF。**リポジトリは PUBLIC なので git には載せない**（`.gitignore` 済み）
- 実験の結果を卒論に使うときは、`results/` や `git show exp/<ブランチ>:<パス>` で読んで `docs/thesis/` に図表を起こし、main にコミットする
