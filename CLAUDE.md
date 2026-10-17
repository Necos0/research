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
- 実験固有の文書（README の「現在の実験」節、`docs/results_*.html` など）は、従来どおり各 `exp/*` ブランチに置く。
