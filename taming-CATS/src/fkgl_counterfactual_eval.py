"""FKGL の目標値を書き換えた推論結果（cf_<条件>/output_1.json）を、与えた目標に対して集計する（Mac 側で実行）。

パイプラインの MAE は参照文の FKGL を目標にして計算されるので、この実験では使えない。
ここでは各出力のプロンプトから「モデルに実際に与えた目標」を読み取り、それに対して集計する。

条件ごと:
- 目標の平均 / 出力 FKGL の平均 / MAE（与えた目標に対して）
- コピー率、数値の保持（原文に数値がある事例。ミクロ・完全保持）
条件の組（abs* と ref*）ごと:
- 傾き: 文ごとに平均を引いたうえでの「出力 FKGL ~ 目標」の回帰係数（1 なら目標どおりに動く、0 なら無視）
- 文ごとの順位相関（Spearman）の平均と、目標の順に出力が厳密に並ぶ文の割合

使い方: python src/fkgl_counterfactual_eval.py <output/sft_inference/<model> のディレクトリ>
"""
import json
import os
import re
import sys

import numpy as np

from build_fkgl_counterfactual_tests import ABSOLUTE, RELATIVE
from helpers.keep import extract_numbers

TARGET_RE = re.compile(r"EXPLANATION: The <FKGL=([-\d.]+)>")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def load(path):
    items = json.load(open(path))
    targets = []
    for x in items:
        m = TARGET_RE.search(x["prompt"])
        assert m, f"プロンプトに目標が見つからない: {x['global_id']}"
        targets.append(float(m.group(1)))
    outputs = np.array([x["prediction_metrics"]["FKGL"] for x in items])
    return items, np.array(targets), outputs


def condition_stats(items, targets, outputs):
    copy = [norm(x["prediction"]) == norm(x["source_text"]) for x in items]
    kept = total = complete = numeric = 0
    for x in items:
        src = extract_numbers(x["source_text"])
        if not src:
            continue
        pred = set(extract_numbers(x["prediction"]))
        k = sum(n in pred for n in src)
        kept, total, complete, numeric = kept + k, total + len(src), complete + (k == len(src)), numeric + 1
    return {
        "target": targets.mean(),
        "output": outputs.mean(),
        "mae": np.abs(outputs - targets).mean(),
        "copy": np.mean(copy),
        "kept": f"{kept}/{total}",
        "complete": f"{complete}/{numeric}",
    }


def rank(a):
    return np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)


def group_stats(results, conds):
    """同じ文に複数の目標を与えたときに、出力が目標に従って動くか。"""
    T = np.column_stack([results[c][1] for c in conds])  # 文 × 条件
    Y = np.column_stack([results[c][2] for c in conds])
    Tc, Yc = T - T.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    slope = (Tc * Yc).sum() / (Tc * Tc).sum()
    rhos, ordered = [], 0
    for t, y in zip(T, Y):
        rt, ry = rank(t), rank(y)
        if ry.std() > 0:
            rhos.append(np.corrcoef(rt, ry)[0, 1])
        ordered += bool(np.all(np.diff(y[np.argsort(t)]) > 0))
    return {"slope": slope, "rho": np.mean(rhos), "ordered": ordered / len(T), "n": len(T)}


def main():
    root = sys.argv[1]
    conds = [c for c in list(ABSOLUTE) + list(RELATIVE) if os.path.exists(os.path.join(root, f"cf_{c}", "output_1.json"))]
    results = {c: load(os.path.join(root, f"cf_{c}", "output_1.json")) for c in conds}
    ids = [x["global_id"] for x in results[conds[0]][0]]
    for c in conds:  # 行の順番で突き合わせる（test には global_id の重複が1つある）
        assert [x["global_id"] for x in results[c][0]] == ids, f"{c} の行の並びが違う"

    print("| 条件 | 目標の平均 | 出力 FKGL の平均 | MAE（与えた目標に対して） | コピー率 | 数値の保持 | 完全保持 |")
    print("|---|---|---|---|---|---|---|")
    for c in conds:
        s = condition_stats(*results[c])
        print(f"| {c} | {s['target']:.2f} | {s['output']:.2f} | {s['mae']:.2f} | {s['copy']:.1%} | {s['kept']} | {s['complete']} |")
    print()
    print("| 組 | 傾き（文ごとに平均を引いた回帰） | 文ごとの順位相関の平均 | 目標の順に厳密に並ぶ文 |")
    print("|---|---|---|---|")
    for name, group in [("abs", list(ABSOLUTE)), ("ref", list(RELATIVE))]:
        group = [c for c in group if c in results]
        if len(group) >= 2:
            g = group_stats(results, group)
            print(f"| {name}（{', '.join(group)}） | {g['slope']:.2f} | {g['rho']:.2f} | {g['ordered']:.1%}（{g['n']}文中） |")


if __name__ == "__main__":
    main()
