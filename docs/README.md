# taming-CATS コードベース解説（docs インデックス）

`taming-CATS/` は、**制御トークン (control tokens) を用いた制御可能テキスト平易化
(Controllable Automatic Text Simplification, CATS)** の論文実装です。
LLM を教師ありファインチューニング (SFT) し、`<FKGL=6>` のような数値トークンで
「6年生レベルに平易化」「文字数を 0.7 倍に圧縮」といった**目標値を明示的に指定して生成を制御**できるようにします。

このドキュメント群は、コードを読み解いた上で

1. **各ディレクトリ・ファイルが何をしているのか**
2. **自分で再現実験をする場合にどこを直せばよいのか**

を整理したものです。

## ドキュメント一覧

| ファイル | 内容 |
| --- | --- |
| [experiment_flow.md](experiment_flow.md) | データ分割→学習→推論→評価の流れを mermaid 図でまとめたもの（実験の全体像把握用） |
| [directory_roles.md](directory_roles.md) | リポジトリ内の各ディレクトリ・主要スクリプトの役割を網羅的に説明 |
| [reproduction_guide.md](reproduction_guide.md) | 再現実験の手順と「どこを書き換えるべきか」の具体的ガイド |

> 補足: `taming-CATS/documents/` にも著者による詳細な設計ノート（データ前処理・制御属性・プロンプト設計・WikiLarge 処理履歴）があります。
> 本 docs はそれらを踏まえつつ「コードのどこを触るか」という再現実験者目線で再構成したものです。

## 全体パイプライン（5段階）

```
[1] データ準備                [2] SFT学習            [3] 推論              [4] 評価            [5] 可視化
─────────────────────       ──────────────       ──────────────       ──────────────      ──────────────
生データ                                                                                 
  │  dataset_to_jsonl.py                                                                  
  ▼                                                                                       
data/datasets/*/dataset.jsonl                                                             
  │  generate_splits.py（外れ値除去＋FKGL層化80/10/10分割）                                  
  ▼                                                                                       
data/splits_new/                                                                          
  │  flatten_splits.py（1ソース複数簡約 → 1行1簡約に平坦化＋間引き）                          
  ▼                                                                                       
data/splits_flattened/                                                                    
  │  filter_flattened_splits.py（FKGL/ARI/Dale-Chall が全て低下する例だけ残す = "_hq"）       
  ▼                                                                                       
data/splits_flattened_filtered/   ──►  sft_finetune.py  ──►  sft_inference.py  ──►  sft_eval.py  ──►  visualize_*.py
  │  create_combined_dataset.py        （制御トークン付き      （制御値を指定して     （要求値 vs 達成値の    （散布図・誤差・
  ▼  （複数データセット結合 = combined）   プロンプトで学習）       簡約文を生成）         相関 / MAE を集計）     比較プロット）
data/splits_flattened_filtered/combined/   models/<実験名>/    output/sft_inference/   output/sft_results/   visualizations/
```

各段階の「どのスクリプト・どの設定を直すか」は [reproduction_guide.md](reproduction_guide.md) を参照してください。

## キーとなる概念

- **制御属性 / 制御トークン**: `<ARI=5>` `<FKGL=6>` `<DALE-CHALL=6>`（可読性系）と
  `<CHAR_COMPRESSION=0.7>` `<WORD_COMPRESSION=0.8>` `<SENTENCE_COMPRESSION=1.5>`（構造/圧縮系）。
  これらは語彙には追加せず、文字単位で分割（compositional tokenization）して扱う。
- **データセット**: Newsela（ニュース）, Med-EASi（医療）, WikiLarge（一般/Wikipedia）, SimPA（行政; 語彙的・統語的）。
  末尾 `_hq` は品質フィルタ済み、`combined` は複数結合版。
- **プロンプト変種**: `no_token` / `token` / `token_explanation` / `token_explanation_examples`
  （説明・例示の量が増える。論文の既定は `token_explanation`）。
- **単一属性 vs 複数属性学習**: `sft_finetune.py`（1 metric 専用）と
  `sft_finetune_all_ctrl_attr.py`（複数 metric をラウンドロビンで混ぜる）。
- **評価指標**: 要求した制御値と実際に達成された値の **Pearson 相関 / MAE / RMSE / 許容範囲内割合**。
