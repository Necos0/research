import os
import json
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

def log_stats(logfile_path, message):
    with open(logfile_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    print(message)

def map_dataset_name(unformatted_name):
    dataset_mapping = {
        "newsela": "Newsela",
        "simpa": "SimPa",
        "medeasi": "Med-EASi",
        "wikilarge_ori_splitwise": "WikiLarge"
    }
    return dataset_mapping.get(unformatted_name)

def filter_metrics(flattened_input_path, filtered_output_path, log_file_path=None, metrics=None,
                   dataset_stats=None, dataset=None, dataset_metric_sets=None):
    total = 0
    kept = 0
    removed = 0
    filtered_data = []

    if metrics is None:
        metrics = ["FKGL"]

    per_metric_removals = defaultdict(int)

    if dataset_metric_sets is not None and dataset not in dataset_metric_sets:
        dataset_metric_sets[dataset] = {m: set() for m in metrics}

    with open(flattened_input_path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            example = json.loads(line.strip())
            source = example.get("source_metrics", {})
            target = example.get("target_metrics", {})
            example_id = hash(json.dumps(example, sort_keys=True))

            failed_metrics = [
                metric for metric in metrics
                if not (
                    source.get(metric) is not None and 
                    target.get(metric) is not None and 
                    target[metric] < source[metric]
                )
            ]

            if not failed_metrics:
                kept += 1
                filtered_data.append(example)
            else:
                removed += 1
                for metric in failed_metrics:
                    per_metric_removals[metric] += 1
                    if dataset_metric_sets is not None:
                        dataset_metric_sets[dataset][metric].add(example_id)

    with open(filtered_output_path, 'w', encoding='utf-8') as f:
        for entry in filtered_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_stats(log_file_path, f"Input file: {flattened_input_path}")
    log_stats(log_file_path, f"Total instances: {total}")
    log_stats(log_file_path, f"Examples kept (all metrics decrease: {metrics}): {kept}")
    log_stats(log_file_path, f"Examples removed: {removed}")
    for metric in metrics:
        log_stats(log_file_path, f"Removed due to {metric} not decreasing: {per_metric_removals[metric]}")
    log_stats(log_file_path, "-" * 50)

    if dataset not in dataset_stats:
        dataset_stats[dataset] = {
            "total": 0,
            "kept": 0,
            "removed": 0,
            "per_metric_removals": defaultdict(int)
        }

    dataset_stats[dataset]["total"] += total
    dataset_stats[dataset]["kept"] += kept
    dataset_stats[dataset]["removed"] += removed

    for metric in metrics:
        dataset_stats[dataset]["per_metric_removals"][metric] += per_metric_removals[metric]

def load_stats(stats_file):
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_data(dataset_stats):
    data = []
    for dataset, stats in dataset_stats.items():
        row = {
            "Dataset": dataset,
            "Total": stats["total"],
            "Kept": stats["kept"],
            "Removed": stats["removed"]
        }
        for metric, count in stats["per_metric_removals"].items():
            row[f"Removed_due_to_{metric}"] = count
        data.append(row)
    return pd.DataFrame(data)

def plot_stats(stats_df, save_dir):
    datasets = stats_df["Dataset"]
    n_datasets = len(datasets)

    fig, axs = plt.subplots(1, 4, figsize=(10, 4))

    if n_datasets == 1:
        axs = [axs]

    for i, dataset in enumerate(datasets):
        ax = axs[i]
        kept = stats_df.loc[i, "Kept"]
        removed = stats_df.loc[i, "Removed"]

        x = 0
        bar_width = 0.6
        ax.bar(x, kept, color="seagreen", width=bar_width, label="Kept", edgecolor="black", alpha=0.7)
        ax.bar(x, removed, bottom=kept, color="orchid", width=bar_width, label="Removed", hatch='//', edgecolor="black", alpha=0.7)

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(
            map_dataset_name(dataset),
            fontsize=16
            )
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # fig.legend(["Kept", "Removed"], loc="upper left", frameon=True, fontsize=10)
    fig.legend(["Kept", "Removed"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=True,
            fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(save_dir, "bars.png"), dpi=300)


def plot_venn_diagrams(dataset_metric_sets, save_dir):
    num_datasets = len(dataset_metric_sets)
    rows = 1
    cols = 4

    fig, axs = plt.subplots(rows, cols, figsize=(10, 3.5))
    axs = axs.flatten()

    for i, (dataset, metric_sets) in enumerate(dataset_metric_sets.items()):
        if len(metric_sets) != 3:
            print(f"Skipping Venn diagram for {dataset} (requires exactly 3 metrics).")
            continue

        sets = list(metric_sets.values())
        labels = list(metric_sets.keys())

        ax = axs[i]
        v = venn3(sets, set_labels=labels, ax=ax, alpha=0.5)

        for label in v.set_labels:
            if label:
                label.set_fontsize(12)

        for label in v.subset_labels:
            if label:
                label.set_fontsize(12)

        ax.set_title(f"{map_dataset_name(dataset)}", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "venn.png"), dpi=300)

def main():
    datasets = [
                "medeasi",
                "simpa",
                "wikilarge_ori_splitwise",
                "newsela",
                # "wikilarge_ori_global"
                ]
    metrics_to_filter_by = ["FKGL", "ARI", "Dale-Chall"]
    save_dir = "./../data/splits_flattened_filtered"
    os.makedirs(save_dir, exist_ok=True)

    dataset_stats = defaultdict(dict)
    dataset_metric_sets = {}  # <- for Venn diagrams

    for dataset in datasets:
        print(f"{5 * '*'} Filtering {dataset}... {5 * '*'}")
        input_dir = f"./../data/splits_flattened/{dataset}"
        output_dir = f"{save_dir}/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        log_file_path = os.path.join(output_dir, "log.txt")
        open(log_file_path, "w").close()

        for split in ["train", "val", "test"]:
            input_path = f"{input_dir}/{split}.jsonl"
            output_path = f"{output_dir}/{split}.jsonl"
            filter_metrics(
                input_path,
                output_path,
                log_file_path=log_file_path,
                metrics=metrics_to_filter_by,
                dataset_stats=dataset_stats,
                dataset=dataset,
                dataset_metric_sets=dataset_metric_sets
            )

        log_stats(log_file_path, f"Filtering completed for dataset: {dataset}\n")

    with open(f"{save_dir}/stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=4)

    print(f"Filtering completed for all datasets. Stats saved in '{save_dir}/stats.json'.")

    stats_df = prepare_data(dataset_stats)
    plot_stats(stats_df, save_dir)
    print(f"Stats plots saved in '{save_dir}'.")
    plot_venn_diagrams(dataset_metric_sets, save_dir)
    print(f"Venn diagrams saved in '{save_dir}'.")

if __name__ == "__main__":
    main()
