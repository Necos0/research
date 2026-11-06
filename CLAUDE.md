# CLAUDE.md

このリポジトリで作業を始める前に、必ず次の2つを読み込んでから着手すること。

1. [`docs/direction.md`](docs/direction.md) — 研究の背景・目的・課題・提案手法・実験の実行フロー（共通運用）
2. [`docs/roadmap.md`](docs/roadmap.md) — 次の中間発表に向けたロードマップ（進捗チェックリスト）。1回目の中間発表は終了済み

これらが研究の方向性と現在地を定義する。タスクに取りかかる際は、まず両ファイルを参照して文脈を把握すること。

## 共通文書は main が正本（どのブランチにいても main 版を読む・書く）

`CLAUDE.md`・`docs/direction.md`・`docs/roadmap.md` は全実験に共通の文書なので、**main ブランチだけが正本**。
main は常に worktree **`/Users/wadaketsunin/research-main`** に展開してある。

- **読むとき**: 今どの `exp/*` ブランチにいても、`/Users/wadaketsunin/research-main/docs/` の版を読む。先に `git -C /Users/wadaketsunin/research-main pull --ff-only` で最新にする。
- **書くとき**: `/Users/wadaketsunin/research-main` 側を編集し、**main にコミット**する（push はユーザーの了承を得てから）。`exp/*` ブランチ側の共通文書は編集しない。
- **`exp/*` ブランチにある共通文書は、main を取り込んだ時点の写し**で、古いことがある。読まない・編集しない。
- **実験ブランチへは `git merge main` で取り込む**。main のコードは古いので、実験ブランチは従来どおり元の実験ブランチから切る。切ったら `git merge main` で共通文書とこの CLAUDE.md を入れる。main には共通文書のコミットしか積まないので、このマージで変わるのは共通文書だけ。衝突したら main 側を採用する（`git checkout --theirs CLAUDE.md docs/direction.md docs/roadmap.md`）。
- worktree が無い場合は作り直す: `git -C /Users/wadaketsunin/research worktree add /Users/wadaketsunin/research-main main`
- **`/Users/wadaketsunin/research` 自体が main にいるときは worktree は不要**（同じブランチは2か所でチェックアウトできないので作れない）。そのまま `docs/` を読み書きする。`exp/*` に切り替えたら、上のコマンドで worktree を作り直す。
- 実験固有の文書（README の「現在の実験」節、`docs/results_*.html` など）は、従来どおり各 `exp/*` ブランチに置く。

## 卒論も main が正本（実験が終わっても main に残す）

卒論（LaTeX）と卒論メモは全実験に共通なので、共通文書と同じく **main だけが正本**。必ず `/Users/wadaketsunin/research-main` 側で書き、main にコミットする。`git merge main` で `exp/*` に入った写しは読まない・編集しない。

- `thesis/` — 卒論本体。研究室テンプレート（inouelab-waseda/tex-template）由来、upLaTeX → biber → dvipdfmx。VS Code で `/Users/wadaketsunin/research-main/thesis` を開き、`tex/sample.tex` を保存すると `tex/.latexmkrc` と `.vscode/settings.json` で自動ビルドされる
- `docs/卒論.md`（先生の指示）、`docs/卒論_構成案.md`（章立て）
- `docs/卒論_先輩/` — 先輩の卒論 PDF。**リポジトリは PUBLIC なので git には載せない**（`.gitignore` 済み）
- 実験の結果を卒論に使うときは、`exp/*` 側の結果を読んで `thesis/` に図表を起こし、main にコミットする
