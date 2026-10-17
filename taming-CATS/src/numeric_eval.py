"""数値情報の保持と捏造を、複数の推論出力（output_*.json）について横並びで集計する。

KEEP と FKGL を同じ test 203 件で比較するための評価（ロードマップ③-7・③-8）。
数値の抽出は学習データ付与・既存評価と同じ helpers.keep.extract_numbers を使う。

出す指標:
- 保持（原文に数値がある事例のみ）: マクロ平均 / ミクロ（何個中何個）/ 完全保持事例率
- 非コピー出力での保持: 原文をそのまま写した出力は自動的に全保持になるので、それを除いた保持率
- 捏造: 原文に無い数値を出力に含む事例数（全事例）。参照文の同じ値も併記する
- コピー率: 完全コピー（原文と一致）の割合（全事例・数値あり事例）

使い方:
    python src/numeric_eval.py FKGL-ep2=path/to/output_1.json KEEP-ep2=path/to/output_1.json ...
"""
import json
import re
import sys

from helpers.keep import extract_numbers


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def is_copy(x):
    return norm(x["prediction"]) == norm(x["source_text"])


def retention_stats(items):
    """原文に数値がある事例について保持率を集計する。"""
    per_item, kept, total, complete = [], 0, 0, 0
    for x in items:
        src = extract_numbers(x["source_text"])
        pred = set(extract_numbers(x["prediction"]))
        k = sum(n in pred for n in src)
        per_item.append(k / len(src))
        kept += k
        total += len(src)
        complete += k == len(src)
    n = len(items)
    return {
        "n": n,
        "macro": sum(per_item) / n if n else float("nan"),
        "micro": f"{kept}/{total}",
        "micro_rate": kept / total if total else float("nan"),
        "complete": f"{complete}/{n}",
    }


def hallucinated(text, source_text):
    src = set(extract_numbers(source_text))
    return [n for n in extract_numbers(text) if n not in src]


def evaluate(items):
    numeric = [x for x in items if extract_numbers(x["source_text"])]
    non_copy_numeric = [x for x in numeric if not is_copy(x)]
    hall_pred = [x for x in items if hallucinated(x["prediction"], x["source_text"])]
    hall_ref = [x for x in items if hallucinated(x["reference_simplification"], x["source_text"])]
    return {
        "all": retention_stats(numeric),
        "non_copy": retention_stats(non_copy_numeric),
        "copy_all": sum(map(is_copy, items)) / len(items),
        "copy_numeric": sum(map(is_copy, numeric)) / len(numeric),
        "hall_pred": len(hall_pred),
        "hall_ref": len(hall_ref),
        "hall_examples": [(x["global_id"], hallucinated(x["prediction"], x["source_text"])) for x in hall_pred],
        "n": len(items),
    }


def main():
    rows = []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        rows.append((label, evaluate(json.load(open(path)))))

    print("| 条件 | 保持 マクロ | 保持 ミクロ | 完全保持 | 非コピーでの保持 ミクロ（件数） | コピー率 全体 / 数値あり | 数値の捏造（出力 / 参照文） |")
    print("|---|---|---|---|---|---|---|")
    for label, r in rows:
        a, nc = r["all"], r["non_copy"]
        print(f"| {label} | {a['macro']:.3f} | {a['micro']} ({a['micro_rate']:.1%}) | {a['complete']} "
              f"| {nc['micro']} ({nc['micro_rate']:.1%}, n={nc['n']}) "
              f"| {r['copy_all']:.1%} / {r['copy_numeric']:.1%} | {r['hall_pred']} / {r['hall_ref']} (全{r['n']}件中) |")
    print()
    for label, r in rows:
        if r["hall_examples"]:
            print(f"[{label}] 捏造された数値: " + "; ".join(f"{gid}:{','.join(ns)}" for gid, ns in r["hall_examples"]))


if __name__ == "__main__":
    main()
