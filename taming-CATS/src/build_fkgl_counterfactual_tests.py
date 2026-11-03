"""FKGL の目標値（出力側のタグ）だけを書き換えたテストデータを作る（タグを書き換える実験・推論のみ）。

学習済みの FKGL モデルが、参照文の FKGL 以外の目標を与えられても、タグに従って
出力の難しさを変えるか（介入による制御の確認）を見る。
原文・参照文・原文側のタグ（source_metrics.FKGL）はそのままで、target_metrics.FKGL
（出力側のタグとプロンプトの EXPLANATION に入る値）だけを書き換える。

- abs4 / abs8 / abs12 / abs16 : 全事例に同じ目標（4, 8, 12, 16）。同じ文が目標の順に並ぶかを見る
- refm4 / refm2 / ref0 / refp2: 参照文の FKGL から -4, -2, 0, +2。各文にとって現実的な範囲で従うかを見る
  （ref0 は書き換えなし＝従来の結果の再現の確認用）

load_dataset_from_hf は train/val/test の3ファイルを読むので、train/val はフル版をそのまま写す。
列を足すと3ファイルの列がそろわず datasets の読み込みが失敗するので、タグの値以外は変えない。
出力先は gitignore 済みの output/ 配下（run_experiment.sh の掃除で消える。このスクリプトで再生成できる）。

使い方: python src/build_fkgl_counterfactual_tests.py [出力ルート]
"""
import json
import os
import shutil
import sys

SRC = os.path.join("data", "splits_flattened_full", "medeasi")
ABSOLUTE = {"abs4": 4.0, "abs8": 8.0, "abs12": 12.0, "abs16": 16.0}
RELATIVE = {"refm4": -4.0, "refm2": -2.0, "ref0": 0.0, "refp2": 2.0}
CONDITIONS = list(ABSOLUTE) + list(RELATIVE)


def target_value(condition, row):
    reference = row["target_metrics"]["FKGL"]
    if condition in ABSOLUTE:
        return ABSOLUTE[condition]
    if condition == "ref0":
        return reference
    if condition in RELATIVE:
        return round(reference + RELATIVE[condition], 1)
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
                value = target_value(cond, row)
                changed += value != row["target_metrics"]["FKGL"]
                row["target_metrics"]["FKGL"] = value
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{cond:6s}: {len(rows)} rows, target changed in {changed} -> {dst}")


if __name__ == "__main__":
    main()
