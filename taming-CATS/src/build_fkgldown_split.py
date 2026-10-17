"""フル版スプリットから「参照文の FKGL が原文より下がっていない」事例を train / val から除く。

コピー崩壊の切り分け②（データ）。フル版 train の 34.2% は参照文が易化されておらず
（FKGL が上昇 27.6%・同じ 6.6%）、「易しくしない」ことを教えている。FKGL だけに焦点を
当てるため、可読性フィルタ（FKGL・ARI・Dale-Chall すべて低下）ではなく FKGL のみで判定する。

- train / val: target_metrics.FKGL < source_metrics.FKGL の事例だけ残す
- test: フル版の 203 件をそのまま使う（全条件で同じテストにして比較できるようにする）

使い方: python src/build_fkgldown_split.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
SRC = os.path.join(ROOT, "splits_flattened_full", "medeasi")
DST = os.path.join(ROOT, "splits_flattened_full_fkgldown", "medeasi")


def main():
    os.makedirs(DST, exist_ok=True)
    log = ["Input: data/splits_flattened_full/medeasi",
           "Rule: train/val keep only target FKGL < source FKGL; test copied unchanged"]
    for split in ["train", "val", "test"]:
        with open(os.path.join(SRC, f"{split}.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        if split == "test":
            kept = rows
        else:
            kept = [r for r in rows if r["target_metrics"]["FKGL"] < r["source_metrics"]["FKGL"]]
        with open(os.path.join(DST, f"{split}.jsonl"), "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        msg = f"{split}: {len(rows)} -> {len(kept)} rows (removed {len(rows) - len(kept)})"
        print(msg)
        log.append(msg)
    with open(os.path.join(DST, "log.txt"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
