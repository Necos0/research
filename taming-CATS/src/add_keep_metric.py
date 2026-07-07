"""整形済みデータに <KEEP> 制御トークン用のメトリックを追記する（ロードマップ2.0）。

source_metrics["keep"] / target_metrics["keep"] に保持すべき数値の整形済み
文字列を書き込む。タグと教師信号を整合させるため、値はスプリットで使い分ける
（build_keep_full_splits.py と同じ規則）:
- train / val: 原文の数値のうち正解文が実際に保持している数値
  （1つも保持されていなければ "none" ＝保持制約なしの教師）
- test: 原文の全数値（「全部保持せよ」という指示に従えるかを評価する）
既存の FKGL 等のキーには一切触れない（追記のみ）。

使い方:
    python src/add_keep_metric.py --dataset medeasi
    （data/splits_flattened_filtered/<dataset>/{train,val,test}.jsonl を上書き更新）
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.keep import keep_value_from_source, keep_value_from_pair, extract_numbers


def augment_file(path, split):
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    rows_with_numbers = 0
    rows_with_tag = 0
    for row in rows:
        source_text = row.get("source_text", "")
        if split == "test":
            keep_value = keep_value_from_source(source_text)
        else:
            keep_value = keep_value_from_pair(source_text, row.get("simplification_text", ""))
        if extract_numbers(source_text):
            rows_with_numbers += 1
        if keep_value != "none":
            rows_with_tag += 1
        row.setdefault("source_metrics", {})["keep"] = keep_value
        row.setdefault("target_metrics", {})["keep"] = keep_value

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{path}: {len(rows)} rows updated ({rows_with_numbers} contain numbers, {rows_with_tag} with keep tag != none)")


def main():
    parser = argparse.ArgumentParser(description="Add <KEEP> metric to flattened splits.")
    parser.add_argument("--dataset", type=str, default="medeasi", help="Dataset folder name.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/splits_flattened_filtered",
        help="Root dir of flattened, filtered splits.",
    )
    args = parser.parse_args()

    base = os.path.join(args.data_dir, args.dataset)
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] Missing split file: {path}")
            continue
        augment_file(path, split)

    print("Done adding <KEEP> metric.")


if __name__ == "__main__":
    main()
