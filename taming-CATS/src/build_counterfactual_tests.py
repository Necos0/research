"""KEEP タグの値だけを書き換えたテストデータを作る（タグを書き換える実験・推論のみ）。

学習済みの KEEP モデルが、タグの指示に従って数値を残す／落とすかを確かめる。
原文・参照文はそのままで、source_metrics.keep と target_metrics.keep（原文側と出力側の
タグ）だけを書き換える。

- normal : 原文の全数値（従来どおり。再現の確認用）
- none   : 数値がある事例も "none"（保持の指示を外す）
- partial: 数値が2つ以上ある事例は最初の数値だけを指定する（それ以外の事例は normal と同じ）
- fake   : 全事例に、原文にない数値（DISTRACTORS の先頭から原文に無いもの）を1つ加える

load_dataset_from_hf は train/val/test の3ファイルを読むので、train/val はフル版をそのまま写す。
出力先は gitignore 済みの output/ 配下（run_experiment.sh の掃除で消える。このスクリプトで再生成できる）。

使い方: python src/build_counterfactual_tests.py [出力ルート]
"""
import json
import os
import shutil
import sys

from helpers.keep import extract_numbers, format_keep, NO_NUMBERS

SRC = os.path.join("data", "splits_flattened_full", "medeasi")
DISTRACTORS = ["37", "73", "91"]
CONDITIONS = ["normal", "none", "partial", "fake"]


def distractor_for(text):
    present = set(extract_numbers(text))
    for d in DISTRACTORS:
        if d not in present and d not in text:
            return d
    raise ValueError(f"使える囮の数値がない: {text}")


def keep_value(condition, row):
    nums = extract_numbers(row["source_text"])
    if condition == "normal":
        return format_keep(nums)
    if condition == "none":
        return NO_NUMBERS
    if condition == "partial":
        return format_keep(nums[:1]) if len(nums) >= 2 else format_keep(nums)
    if condition == "fake":
        return format_keep(nums + [distractor_for(row["source_text"])])
    raise ValueError(condition)


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("output", "counterfactual_data")
    with open(os.path.join(SRC, "test.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    for cond in CONDITIONS:
        dst = os.path.join(out_root, cond, "medeasi")
        os.makedirs(dst, exist_ok=True)
        for split in ["train", "val"]:
            shutil.copyfile(os.path.join(SRC, f"{split}.jsonl"), os.path.join(dst, f"{split}.jsonl"))
        changed = 0
        with open(os.path.join(dst, "test.jsonl"), "w") as f:
            for row in rows:
                row = json.loads(json.dumps(row))
                value = keep_value(cond, row)
                changed += value != row["source_metrics"]["keep"]
                row["source_metrics"]["keep"] = value
                row["target_metrics"]["keep"] = value
                row["counterfactual_condition"] = cond
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{cond:8s}: {len(rows)} rows, tag changed in {changed} -> {dst}")


if __name__ == "__main__":
    main()
