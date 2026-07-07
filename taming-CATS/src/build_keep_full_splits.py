"""フル（未フィルタ）データから <KEEP> 実験用のスプリットを作成する。

splits_flattened_filtered は、FKGL 等の外れ値除去（generate_splits.py）と
「全可読性指標が下がる事例のみ残す」フィルタ（filter_flattened_splits.py）で
Med-EASi 1892 事例 → 852 件まで減っている。<KEEP>（数値保持）実験では可読性の
フィルタは不要なので、data/datasets/<dataset>/dataset.jsonl の全事例を使う。

スプリット割り当ては既存の splits_flattened_filtered と整合させる:
- 既存スプリットに含まれる global_id は同じスプリットへ割り当てる
  （旧 test ⊂ 新 test となり、フィルタ済み実験と同一部分集合での比較評価が可能。
    旧 train の事例が新 test に混ざるリークも防ぐ）
- 残り（フィルタで落ちていた事例）は 80/10/10 で割り当てる。その際
  「数値なし／数値ありで正解が保持／数値ありで正解が全滅」の3グループの割合が
  全スプリットで全体比率に揃うよう、グループ別にスプリット・クォータを計算し
  （引き継ぎ分の偏りもここで補正する）、各グループ内は FKGL 順に均等配分する
  （FKGL 分布の層化も維持。seed 42）。これにより「原文に数値がある割合」と
  「train/val で keep タグが付く割合」の両方が揃う。

各行には helpers/keep.py のロジックで source_metrics["keep"] /
target_metrics["keep"] を付与する。タグと教師信号を整合させるため、値は
スプリットで使い分ける（FKGL タグが学習時に「正解文の実際の値」を使うのと同じ原則）:
- train / val: 原文の数値のうち正解文が実際に保持している数値
  （1つも保持されていなければ "none" ＝保持制約なしの教師）
- test: 原文の全数値（「全部保持せよ」という指示に従えるかを評価する）

使い方（taming-CATS ディレクトリから）:
    python src/build_keep_full_splits.py --dataset medeasi
    （data/splits_flattened_full/<dataset>/{train,val,test}.jsonl を生成）
"""

import os
import re
import sys
import json
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers.keep import keep_value_from_source, keep_value_from_pair, extract_numbers


SPLITS = ["train", "val", "test"]
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42

# 層化グループ: 数値なし / 数値ありで正解が1つ以上保持 / 数値ありで正解が全滅
GROUPS = ["no_numbers", "retained", "lost"]


def clean_text(text):
    """flatten_splits.py と同一の整形（改行・連続空白の正規化）。"""
    if not isinstance(text, str):
        return text
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def log_stats(log_file_path, message):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)


def load_existing_split_map(filtered_dir):
    """既存の splits_flattened_filtered から global_id → split の対応を作る。"""
    split_map = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(filtered_dir, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        for row in load_jsonl(path):
            global_id = row["global_id"]
            if split_map.get(global_id, split) != split:
                raise ValueError(f"global_id {global_id} が複数スプリットに存在します")
            split_map[global_id] = split
    return split_map


def apportion(total, weights):
    """total 件を weights に比例配分する（最大剰余法。合計は必ず total）。"""
    keys = list(weights)
    weight_sum = sum(weights.values())
    ideals = {k: total * weights[k] / weight_sum for k in keys}
    counts = {k: int(ideals[k]) for k in keys}
    leftover = total - sum(counts.values())
    for k in sorted(keys, key=lambda k: (-(ideals[k] - counts[k]), keys.index(k)))[:leftover]:
        counts[k] += 1
    return counts


def deal_by_fkgl(examples, quotas, rng):
    """FKGL 順に並べた事例をクォータ比で各スプリットへ順繰りに配る。

    FKGL の小さい順に見て「次の1件で quota 超過率が最も小さいスプリット」へ
    割り当てる（D'Hondt 式）。各スプリットが FKGL 全域から均等に取るため
    FKGL 分布の層化になり、かつクォータをちょうど満たす。
    """
    assignment = {}
    ordered = sorted(examples, key=lambda ex: (ex["source_metrics"].get("FKGL", 0), rng.random()))
    assigned = {s: 0 for s in SPLITS}
    for ex in ordered:
        candidates = [s for s in SPLITS if assigned[s] < quotas[s]]
        split = min(candidates, key=lambda s: ((assigned[s] + 1) / quotas[s], SPLITS.index(s)))
        assignment[ex["global_id"]] = split
        assigned[split] += 1
    return assignment


def example_group(example):
    """層化グループを返す（no_numbers / retained / lost）。

    retained: 原文に数値があり、いずれかの正解文が1つ以上保持している
    （＝train/val で keep タグ != none になる事例）。
    """
    source_text = clean_text(example.get("source_text", ""))
    numbers = extract_numbers(source_text)
    if not numbers:
        return "no_numbers"
    for simplification in example.get("simplifications", []) or []:
        target_numbers = set(extract_numbers(clean_text(simplification.get("simplification_text", ""))))
        if any(n in target_numbers for n in numbers):
            return "retained"
    return "lost"


def balanced_assign(unassigned, inherited_sizes, inherited_group_counts, seed=SEED):
    """未割り当て事例を 80/10/10 に割り当てる（層化グループ×FKGL）。

    引き継ぎ分も含めた最終スプリットで各グループ（数値なし／正解が保持／
    正解が全滅）の割合が全体比率に揃うよう、グループ別にスプリット・クォータを
    決める（引き継ぎ分の偏りをここで補正）。
    """
    assignment = {}
    if not unassigned:
        return assignment

    grouped = {g: [] for g in GROUPS}
    for ex in unassigned:
        grouped[example_group(ex)].append(ex)

    # 最終スプリットサイズ = 引き継ぎ分 + 未割り当て分の 80/10/10
    new_sizes = apportion(len(unassigned), SPLIT_FRACTIONS)
    final_sizes = {s: inherited_sizes[s] + new_sizes[s] for s in SPLITS}

    # retained / lost のクォータ: グループ総数（引き継ぎ分含む）をスプリット
    # サイズに比例配分し、引き継ぎ分を差し引く。no_numbers は残り。
    quotas = {}
    for group in ["retained", "lost"]:
        total = len(grouped[group]) + sum(inherited_group_counts[group].values())
        targets = apportion(total, final_sizes)
        quotas[group] = {s: targets[s] - inherited_group_counts[group][s] for s in SPLITS}
        if any(q < 0 or q > new_sizes[s] for s, q in quotas[group].items()):
            raise ValueError(f"グループ {group} のクォータが計算できません: {quotas[group]}")
    quotas["no_numbers"] = {
        s: new_sizes[s] - quotas["retained"][s] - quotas["lost"][s] for s in SPLITS
    }
    if any(q < 0 for q in quotas["no_numbers"].values()):
        raise ValueError(f"グループ no_numbers のクォータが計算できません: {quotas['no_numbers']}")

    rng = random.Random(seed)
    for group in GROUPS:
        assignment.update(deal_by_fkgl(grouped[group], quotas[group], rng))
    return assignment


def flatten_example(example, split):
    """flatten_splits.py と同じ形の行（1平易化=1行）に展開し、keep を付与する。

    keep の値: train / val は正解文が実際に保持している数値（教師信号と整合）、
    test は原文の全数値（保持指示）。
    """
    rows = []
    source_text = clean_text(example.get("source_text", ""))

    metadata = dict(example.get("metadata", {}))
    metadata.pop("original_split", None)  # generate_splits.py と同様に除去

    for simplification in example.get("simplifications", []) or []:
        simplification_text = clean_text(simplification.get("simplification_text", ""))
        if split == "test":
            keep_value = keep_value_from_source(source_text)
        else:
            keep_value = keep_value_from_pair(source_text, simplification_text)

        source_metrics = dict(example.get("source_metrics", {}))
        target_metrics = dict(simplification.get("target_metrics", {}))
        source_metrics["keep"] = keep_value
        target_metrics["keep"] = keep_value
        rows.append({
            "global_id": example["global_id"],
            "source_text": source_text,
            "source_metrics": source_metrics,
            "metadata": metadata,
            "simplification_text": simplification_text,
            "target_metrics": target_metrics,
            "simplification_version": simplification.get("simplification_version", ""),
            "grade_level": simplification.get("grade_level", ""),
            "simplification_dimensions": simplification.get("simplification_dimensions", {}),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build unfiltered full-data splits with <KEEP> metric."
    )
    parser.add_argument("--dataset", type=str, default="medeasi")
    parser.add_argument("--dataset_dir", type=str, default="data/datasets")
    parser.add_argument(
        "--filtered_dir",
        type=str,
        default="data/splits_flattened_filtered",
        help="スプリット割り当てを引き継ぐ既存のフィルタ済みスプリット。",
    )
    parser.add_argument("--output_dir", type=str, default="data/splits_flattened_full")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "dataset.jsonl")
    examples = load_jsonl(dataset_path)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = os.path.join(out_dir, "log.txt")
    open(log_file_path, "w").close()

    log_stats(log_file_path, f"Input dataset: {dataset_path} ({len(examples)} examples)")

    # 1) 既存スプリットの割り当てを引き継ぐ
    split_map = load_existing_split_map(os.path.join(args.filtered_dir, args.dataset))
    log_stats(
        log_file_path,
        f"Split assignments inherited from {args.filtered_dir}: {len(split_map)} examples",
    )

    # 2) 残り（フィルタで落ちていた事例）を、グループ比率が揃うよう層化して割り当てる
    unassigned = [ex for ex in examples if ex["global_id"] not in split_map]
    inherited_sizes = {s: 0 for s in SPLITS}
    inherited_group_counts = {g: {s: 0 for s in SPLITS} for g in GROUPS}
    for ex in examples:
        split = split_map.get(ex["global_id"])
        if split is None:
            continue
        inherited_sizes[split] += 1
        inherited_group_counts[example_group(ex)][split] += 1
    split_map.update(balanced_assign(unassigned, inherited_sizes, inherited_group_counts))
    log_stats(
        log_file_path,
        f"Newly assigned (group-ratio balanced [{'/'.join(GROUPS)}], FKGL-stratified, "
        f"seed={SEED}): {len(unassigned)} examples",
    )

    # 3) 展開して keep を付与し、スプリットごとに書き出す
    split_rows = {"train": [], "val": [], "test": []}
    for example in examples:
        split = split_map[example["global_id"]]
        split_rows[split].extend(flatten_example(example, split))

    rng = random.Random(SEED)
    for split in ["train", "val", "test"]:
        rows = split_rows[split]
        rng.shuffle(rows)
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with_numbers = sum(1 for r in rows if extract_numbers(r["source_text"]))
        with_keep_tag = sum(1 for r in rows if r["source_metrics"]["keep"] != "none")
        log_stats(
            log_file_path,
            f"{path}: {len(rows)} rows ({with_numbers} contain numbers "
            f"[{with_numbers / len(rows):.1%}], {with_keep_tag} with keep tag != none)",
        )

    total = sum(len(rows) for rows in split_rows.values())
    log_stats(log_file_path, f"Total flattened rows: {total}")
    log_stats(log_file_path, f"Done building full <KEEP> splits for {args.dataset}.")


if __name__ == "__main__":
    main()
