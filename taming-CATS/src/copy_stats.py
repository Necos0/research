"""推論出力（output_*.json）の原文コピー率を集計する。

SARI / LENS / 制御 MAE は原文コピー崩壊を検出できない（コピー 50% でも SARI は上がった）ため、
評価のたびにこれを併記する。test は全条件で同じ 203 件なので、次の部分集合でも出す:

- ref易化   : 参照文の FKGL が原文より下がっている事例（＝易しくすべき事例）
- ref非易化 : 参照文の FKGL が原文以上の事例（＝コピーしても制御トークン上は誤りでない事例）
- filt98    : フィルタ版（splits_flattened_filtered）の test 98 件（旧実験との比較用）

使い方: python src/copy_stats.py --label epoch1 output/.../output_1.json
"""
import argparse
import difflib
import json
import os
import re

FILTERED_TEST = os.path.join(os.path.dirname(__file__), "..", "data", "splits_flattened_filtered", "medeasi", "test.jsonl")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def summarize(items):
    n = len(items)
    if n == 0:
        return "n=0"
    exact = sum(norm(x["prediction"]) == norm(x["source_text"]) for x in items)
    near = sum(similarity(x["prediction"], x["source_text"]) >= 0.9 for x in items)
    empty = sum(not norm(x["prediction"]) for x in items)
    the = sum(norm(x["prediction"]).startswith("The ") for x in items)
    return (f"n={n:3d}  完全コピー {exact:3d} ({exact / n:5.1%})  類似>=0.9 {near:3d} ({near / n:5.1%})"
            f"  空 {empty}  'The '始まり {the / n:5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_file")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    items = json.load(open(args.output_file))
    with open(FILTERED_TEST) as f:
        filt_ids = {json.loads(line)["global_id"] for line in f}

    def ref_easier(x):
        return x["reference_metrics"]["FKGL"] < x["source_metrics"]["FKGL"]

    print(f"[{args.label}] 全体      {summarize(items)}")
    print(f"[{args.label}] ref易化   {summarize([x for x in items if ref_easier(x)])}")
    print(f"[{args.label}] ref非易化 {summarize([x for x in items if not ref_easier(x)])}")
    print(f"[{args.label}] filt98    {summarize([x for x in items if x['global_id'] in filt_ids])}")


if __name__ == "__main__":
    main()
