# KEEP 用データセットの調査（2026-10-19）

KEEP タグの学習データが少ない問題への対応として、数値を多く含む平易化データセットを調べた。現在の Med-EASi では、train 1499件のうち「残せ」と指示している事例が 249件（16.6%）、数値が 420個しかない。

## まとめ

| データセット | 分野 | 単位 | ライセンス（確認方法） | 文の対応数 | 数値を含む対応 | 参照による数値の保持 | 評価 |
|---|---|---|---|---|---|---|---|
| Med-EASi（現在使用） | 医療 | 文 | 明記なし（GitHub リポジトリは MIT。HF のカードに表記なし） | 1,702 | 389（23%） | 63% | — |
| **PLABA** | 医療（PubMed の抄録） | 文 | **CC0 1.0**（OSF のノードのライセンスを API で確認） | 7,612（1抄録1書き直し）／9,320（全書き直し） | **3,241（43%）** | 49% | **医療の第一候補** |
| **SimPA** | 行政 | 文 | 原典は **CC BY 4.0**（Zenodo）。Taming-CATS 版（HF `shtosti/SimPA`）は MIT 表記 | 2,473（原文 1,237） | 631（26%） | **97%** | **すぐ使える**（今と同じ形式） |
| Cochrane-auto | 医療（Cochrane レビュー） | 文 | GitHub にライセンス表記なし。論文は CC BY 4.0。元文は Cochrane（Wiley）の著作物 | 10,528（train、ignore 除外） | 4,173（40%） | 13%（削除を除くと 21%） | 評価用に限る |
| MultiCochrane | 医療（Cochrane） | 文 | 表記なし。リポジトリは「整備中」 | clean 約5千 | 未計測 | 未計測 | 現時点では使いにくい |
| Cochrane PLS（Devaraj 2021、HF `GEM/cochrane-simplification`） | 医療 | 段落 | HF の表記は CC BY 4.0（元文は Cochrane の著作物） | — | — | — | 段落単位なので今の仕組みに合わない |
| CELLS | 医療（12誌の抄録と一般向け要約） | 段落 | リポジトリは MIT（元文は各誌の著作物） | 62,886 | — | — | 段落単位。将来の拡張向け |

数値の数え方は `helpers/keep.extract_numbers`（学習データにタグを付けるときと同じ）。

## 各データセットの詳細

### PLABA（Plain Language Adaptation of Biomedical Abstracts）
- 出典：Attal, Ondov, Demner-Fushman. *Scientific Data* 2023（[論文](https://www.nature.com/articles/s41597-022-01920-3)、[PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9811873/)）。データは [OSF rnpmf](https://osf.io/rnpmf/)、コードは [GitHub attal-kush/PLABA](https://github.com/attal-kush/PLABA)。
- ライセンス：**OSF 上のデータは CC0 1.0 Universal**。公開リポジトリにも置ける。
- 形式：`data.json` が 75の質問（話題）→ 抄録（PMID ごと）→ `abstract` の文番号ごとの原文と、`adaptations` の文番号ごとの書き直し。文単位で対応している。書き直しが空のものは削除を表す。
- 規模：750抄録、7,612文の対応（1抄録につき書き直し1つを使った場合）。**数値を含む対応は 3,241件で、Med-EASi の約8倍**。
- 参照が落とした数値（4,402個）の内訳（簡易分類）：統計（p 値・信頼区間・オッズ比など）39%、その他 32%、割合 11%、用量・単位 8%、期間・年齢 7%、人数 4%。残した数値（4,259個）は、期間・年齢 21%、人数 14%、割合 13% が多い。**統計の細部は落とし、患者に関わる数値は残す**、という書き直しの方針が見える。KEEP の「残せと指示した数値だけ残す」設計と相性がよい。
- **注意：「その他」には、名前の一部の数字が多い**（"SARS-CoV-2" の 2、"Tsc1" の 1、"ATDC5" の 5 など）。今の `extract_numbers` はこれも数値として拾う。PLABA を使う前に、英字と続いている数字を除くよう抽出処理を直す必要がある（Med-EASi の "GLP-2" も同じ問題）。
- 分割：同じ抄録の文が train と test に分かれないよう、**抄録単位（できれば話題単位）で分ける**必要がある。

### SimPA（Simplification corpus for the Public Administration domain）
- 出典：Scarton, Paetzold, Specia. LREC 2018（[ACL Anthology](https://aclanthology.org/L18-1685/)、[Zenodo](https://zenodo.org/records/1410455)。Zenodo には論文 PDF だけがある）。
- ライセンス：原典は CC BY 4.0。Taming-CATS 著者による整形版 [HF shtosti/SimPA](https://huggingface.co/datasets/shtosti/SimPA) は MIT 表記（列は Med-EASi 版と同じ：`source_text`、`simplification_text`、`source_metrics`、`target_metrics` など）。
- 規模：2,473の対応（原文1,237。1つの原文に複数の書き直しがある）。数値を含む対応は 631件。
- **参照が数値をほぼ全部残す（97%）**。語彙・構文の書き換えが中心で、内容を削らないため。「残せ」の教師信号が多い一方で、もともとコピーに近い書き換えが多いはずなので、コピー率と合わせて見る必要がある。
- `direction.md` の「医療・行政」のうち行政に当たる。今のコードで、`add_keep_metric.py` と分割の作成を通せばほぼそのまま使える見込み。
- 分割：同じ原文の複数の書き直しが train と test に分かれないよう、原文単位で分ける。

### Cochrane-auto
- 出典：Bakker, Kamps. TSAR 2024（[ACL Anthology](https://aclanthology.org/2024.tsar-1.5/)、[GitHub JanB100/cochrane-auto](https://github.com/JanB100/cochrane-auto)）。文・段落・文書単位のデータがある。
- ライセンス：**リポジトリにライセンス表記がない**。論文の記録（Zenodo）は CC BY 4.0。元文は Cochrane のレビューと一般向け要約で、Cochrane（Wiley）の著作物。研究目的での利用は広く行われているが、公開リポジトリにデータ本体を置くのは避けたほうがよい。
- 規模：文単位の train 10,528対応（ignore を除く）。数値を含む対応は 4,173件。
- 参照による数値の保持は 13%（文ごと削除された対応を除いても 21%）。一般向け要約では統計を大胆に省くため。**「全部残せ」の学習には向かない**が、数値の多い難しい評価用データとしては使える。

### MultiCochrane
- 出典：Joseph et al. EMNLP 2023（[論文](https://aclanthology.org/2023.emnlp-main.1037/)、[GitHub SebaJoe/MultiCochrane](https://github.com/SebaJoe/MultiCochrane)）。
- リポジトリは「整備中、追加データはメールで問い合わせ」とあり、ライセンス表記もない。元文は Cochrane。現時点では使いにくい。

### 段落単位のデータ（Cochrane PLS・CELLS）
- [GEM/cochrane-simplification](https://huggingface.co/datasets/GEM/cochrane-simplification)（Devaraj et al. 2021、HF の表記は CC BY 4.0）、[CELLS](https://github.com/LinguisticAnomalies/pls_retrieval)（62,886対応、リポジトリは MIT）。
- どちらも段落・文書単位で、今の文単位の仕組みに合わない。元文は Cochrane や各誌の著作物。

## 今使っている Med-EASi について

- **データ単独のライセンス表記が見つからない**。[GitHub Chandrayee/CTRL-SIMP](https://github.com/chandrayee/ctrl-simp) は MIT だが、データにも適用されるのかは明記されていない。HF の `cbasu/Med-EASi`・`shtosti/Med-EASi` のカードにも表記がない。
- 元文には SimpWiki（Wikipedia 由来）と MSD マニュアル由来の文が含まれる。
- **このリポジトリは公開で、Med-EASi のデータ（`taming-CATS/data/`）をコミットしている**。Taming-CATS 著者も HF で公開しているので実務上の問題は小さいと思われるが、権利関係を確かめたい場合は著者に問い合わせるのが確実。

## 勧める進め方

1. **数値の抽出処理を直す**：英字と続いている数字（"SARS-CoV-2"、"GLP-2"）を除く。年や数値の範囲（"0.3-0.7"）は残す。Med-EASi にも効くので先にやる。
2. **SimPA を追加する**：形式が同じで、CC BY 4.0。手間が最も少ない。
3. **PLABA を追加する**：CC0。数値を含む対応が Med-EASi の約8倍。抄録単位で分割し、今の形式に変換するスクリプトを書く。
4. Cochrane-auto は、必要になったら**評価専用**に使う（データ本体はコミットしない）。

調査で使ったダウンロード物と集計スクリプトは一時フォルダに置いたので残らない。採用するデータセットを決めたら、変換と集計のスクリプトは実験ブランチにコミットする。
