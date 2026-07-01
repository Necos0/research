# 実験フロー全体図（データ分割 → 評価）

この実験の目的は、**制御トークン (`<FKGL=6>` など) を付けて LLM を SFT し、
数値で平易化の目標（可読性レベル・圧縮率）を指定して生成を制御できるか**を検証することにある。
ここでは、データ分割から評価までの流れを mermaid 図で整理する。

---

## 1. 全体パイプライン

```mermaid
flowchart TD
    subgraph PREP["① データ準備"]
        RAW["生データ<br/>Newsela / Med-EASi / WikiLarge / SimPA"]
        JSONL["dataset_to_jsonl.py<br/>統一JSONLスキーマ化<br/>+ 可読性/圧縮率/BLEU/BERTScore 計算"]
        SPLIT["generate_splits.py<br/>外れ値除去(1/99%ile)<br/>FKGL層化 25bin → 80/10/10分割"]
        FLAT["flatten_splits.py<br/>1ソース複数簡約 → 1行1簡約<br/>keep_fraction で間引き"]
        FILT["filter_flattened_splits.py<br/>FKGL/ARI/Dale-Chall が全て低下する例だけ残す<br/>= 高品質版 _hq"]
        COMB["create_combined_dataset.py<br/>複数データセット結合 = combined"]
        RAW --> JSONL --> SPLIT --> FLAT --> FILT --> COMB
    end

    DATA[("data/splits_flattened_filtered/<br/>train / val / test")]
    FILT --> DATA
    COMB --> DATA

    subgraph TRAIN["② SFT学習"]
        FT["sft_finetune.py<br/>制御トークン付きプロンプトで学習<br/>(プロンプト部はlossマスク)"]
    end

    subgraph INFER["③ 推論"]
        INF["sft_inference.py<br/>テストの各文に制御値を指定して生成<br/>複数シードで反復"]
    end

    subgraph EVAL["④ 評価"]
        EV["sft_eval.py<br/>要求値 vs 達成値の<br/>Pearson相関 / MAE / RMSE / 範囲内割合"]
    end

    subgraph VIZ["⑤ 可視化"]
        VS["visualize_*.py<br/>散布図・誤差・MAE比較"]
    end

    DATA -->|train + val| FT
    FT --> MODELS[("models/&lt;実験名&gt;/")]
    MODELS --> INF
    DATA -->|test| INF
    INF --> OUT[("output/sft_inference/<br/>output_i.json")]
    OUT --> EV
    EV --> RESULTS[("output/sft_results/<br/>all_results.json")]
    EV --> VS
    VS --> FIGS[("visualizations/")]

    BASE["run_baseline_inference.sh<br/>未学習モデルで同条件推論"]:::base
    DATA -.->|test| BASE
    BASE -.-> EV
    classDef base fill:#f5f5f5,stroke:#999,stroke-dasharray:5 5;
```

---

## 2. データ分割の詳細（① の内部）

```mermaid
flowchart TD
    A["dataset.jsonl<br/>(1ソース + 複数簡約の階層形式)"]
    A --> B{"外れ値除去<br/>FKGL / ARI / Dale-Chall / 文字数<br/>各1〜99パーセンタイル"}
    B --> C{"件数 > MAX_SAMPLES(3000)?"}
    C -->|Yes| D["ランダムサンプリング<br/>seed=42 で3000件に削減"]
    C -->|No| E["全件保持"]
    D --> F["FKGL を 25bin に層化"]
    E --> F
    F --> G["各binから 80/10/10 抽出<br/>seed=42"]
    G --> TR["train.jsonl"]
    G --> VA["val.jsonl"]
    G --> TE["test.jsonl"]

    TR --> H["flatten_splits.py<br/>1行1簡約に平坦化 + 間引き(keep_fraction=0.4)"]
    VA --> H
    TE --> H
    H --> I{"品質フィルタ<br/>filter_flattened_splits.py<br/>FKGL & ARI & Dale-Chall が全て低下?"}
    I -->|全て低下| J["保持 → _hq 版"]
    I -->|1つでも非低下| K["除外<br/>(自動アラインメント誤り等)"]
```

> WikiLarge の `splitwise`/`global`/`ori` は原 split を保持していない（約2,000件に間引き済みのサブセットから新規分割）。
> 詳細は `taming-CATS/documents/wikilarge_processing_history.md` を参照。

---

## 3. 学習インスタンスの組み立て（② の内部）

制御可能性の核心。プロンプトに制御トークンを埋め込み、**目標値（=簡約文の実測値）を当てるように学習**する。

```mermaid
flowchart TD
    ROW["学習データ1件<br/>source_text / source_metrics / simplification_text / target_metrics"]
    ROW --> SYS["system_prompts.json から<br/>6変種をランダム抽選"]
    ROW --> EXT["制御属性の抽出<br/>可読性系: target側の実測値<br/>圧縮系: target/source の比"]
    EXT --> UP["user_prompts.json から<br/>variant選択 (token_explanation 等)<br/>+ 説明/例を挿入"]
    SYS --> ASM["apply_chat_template で組み立て<br/>(helpers/prompting.py)"]
    UP --> ASM
    ASM --> PROMPT["プロンプト:<br/>SOURCE TEXT: &lt;FKGL=12.3&gt; 元文<br/>... assistant: &lt;FKGL=6&gt;"]
    ASM --> COMP["completion:<br/>簡約文 + EOS"]
    PROMPT --> LOSS["因果LM学習<br/>プロンプト部は -100 でマスク<br/>制御トークン+簡約文+EOS のみlossに寄与"]
    COMP --> LOSS
```

- **単一属性学習** (`sft_finetune.py`): 1モデル = 1 metric。
- **複数属性学習** (`sft_finetune_all_ctrl_attr.py`): metric をラウンドロビンで混在させ 1モデルで複数制御。

---

## 4. 推論 → 評価の流れ（③④ の内部）

```mermaid
flowchart LR
    subgraph I["③ 推論 (sft_inference.py)"]
        direction TB
        T["test の各文"] --> P["ユーザ指定の制御値で<br/>プロンプト構築<br/>例: &lt;FKGL=5&gt;"]
        P --> GEN["生成 (temperature=0 貪欲法)<br/>複数シードで反復"]
        GEN --> M["生成文の実測指標を計算<br/>(classes/Metrics.py)"]
    end
    M --> OUT["output_i.json<br/>(要求値 + 達成値)"]

    subgraph E["④ 評価 (sft_eval.py)"]
        direction TB
        AGG["複数シード集約"] --> CORR["Pearson相関<br/>要求値 vs 達成値"]
        AGG --> ERR["MAE / RMSE / 中央絶対誤差"]
        AGG --> COV["許容範囲内割合 (±10%)"]
    end
    OUT --> AGG
    CORR --> SUM["all_results.json"]
    ERR --> SUM
    COV --> SUM
    SUM --> FIG["散布図(予測vs要求) / 誤差分布 / MAE比較"]
```

**評価の問い**＝「ユーザが要求した制御値 (`<FKGL=5>`) に対し、生成文の実測 FKGL がどれだけ一致するか」。
相関が高く誤差が小さいほど、制御トークンによる平易化制御が効いていることを示す。

---

## 5. 実験で振るパラメータ（比較軸）

```mermaid
mindmap
  root((制御可能な平易化実験))
    モデル
      Llama-3.2-1B/3B
      Llama-3.1-8B
      Qwen3-1.7B/4B/8B
      Ministral-3b / Mistral-7B
    データセット
      Med-EASi 医療
      SimPA 行政
      WikiLarge 一般
      Newsela ニュース
      combined 結合
      _hq 品質フィルタ版
    制御属性
      可読性: ARI / FKGL / Dale-Chall
      圧縮: CHAR / WORD / SENTENCE
    プロンプト変種
      no_token
      token
      token_explanation
      token_explanation_examples
    学習方式
      単一属性
      複数属性 round-robin
      フル FT / PEFT・LoRA
    比較対象
      SFT済みモデル
      ベースライン 未学習
```

各軸を「どのスクリプト・どの設定で変えるか」は [reproduction_guide.md](reproduction_guide.md)、
ディレクトリ・ファイルの役割は [directory_roles.md](directory_roles.md) を参照。
