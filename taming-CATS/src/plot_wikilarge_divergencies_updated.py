import json
import os
import matplotlib.pyplot as plt
import numpy as np

NUM_BINS = 25
DATASET_NAME = "wikilarge_ori"
EXPERIMENT_DIR = f"./../experiments/sample_from_{DATASET_NAME}"
BIN_KEY = f"num_bins_{NUM_BINS}"
DIVERGENCE_FILE = f"{EXPERIMENT_DIR}/all_divergence_results.json"
COLOR_MAP_FILE = "./../data/colormap/color_map.json"
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, BIN_KEY, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DIVERGENCE_FILE, "r") as f:
    ALL_RESULTS = json.load(f)

with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)
METRIC_COLORS = COLOR_MAP["metrics"]

STATS = ["JSD", "EMD", "KS"]
STRAT_TYPES = ["splitwise", "global"]
SEED_KEYS = list(ALL_RESULTS.keys())

first_seed_data = ALL_RESULTS[SEED_KEYS[0]][BIN_KEY]
SUBSET_SIZES = sorted([int(k.split("_")[-1]) for k in first_seed_data.keys()])
SUBSET_KEYS = [f"subset_size_{s}" for s in SUBSET_SIZES]
STRAT_METRICS = [m[0] for m in first_seed_data[SUBSET_KEYS[0]]["splitwise"]["JSD"]]


score_tracker = {
    stat: {
        strat_type: {
            strat_metric: {size: [] for size in SUBSET_SIZES}
            for strat_metric in STRAT_METRICS
        } for strat_type in STRAT_TYPES
    } for stat in STATS
}

for seed_key in SEED_KEYS:
    if BIN_KEY not in ALL_RESULTS[seed_key]:
        continue
    seed_data = ALL_RESULTS[seed_key][BIN_KEY]

    for subset_key in SUBSET_KEYS:
        if subset_key not in seed_data:
            continue
        subset_size = int(subset_key.split("_")[-1])
        entry = seed_data[subset_key]

        for stat in STATS:
            for strat_type in STRAT_TYPES:
                for strat_metric, value in entry[strat_type][stat]:
                    score_tracker[stat][strat_type][strat_metric][subset_size].append(value)

def map_metric_name(metric_ugly):
    metric_mapping = {
        "word_count": "word",
        "sentence_count": "sentence",
        "char_count": "char",
        "FKGL": "FKGL",
        "Dale-Chall": "Dale-Chall",
        "ARI": "ARI"
    }
    return metric_mapping.get(metric_ugly, metric_ugly)

def plot_separately():
    for stat in STATS:
        for strat_type in STRAT_TYPES:
            fig, ax = plt.subplots(figsize=(3.6, 3.6))

            # … your existing plot code …
            for strat_metric in STRAT_METRICS:
                means = [np.mean(score_tracker[stat][strat_type][strat_metric][sz]) for sz in SUBSET_SIZES]
                stds  = [np.std (score_tracker[stat][strat_type][strat_metric][sz]) for sz in SUBSET_SIZES]
                ax.plot(SUBSET_SIZES, means,
                        label=map_metric_name(strat_metric),
                        color=METRIC_COLORS.get(strat_metric,"#000000"))
                ax.fill_between(SUBSET_SIZES,
                                np.array(means)-stds,
                                np.array(means)+stds,
                                alpha=0.2,
                                color=METRIC_COLORS.get(strat_metric,"#000000"))

            ax.set_xlabel("subset size", fontsize=14)
            ax.set_ylabel(f"{stat} score", fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)

            all_means = [np.mean(score_tracker[stat][strat_type][strat_metric][size])
                        for strat_metric in STRAT_METRICS
                        for size in SUBSET_SIZES]
            ymin = max(min(all_means) - 0.005, 0)
            if stat == "KS":
                ymax = 0.05
            elif stat == "JSD":
                ymax = 0.08
            elif stat == "EMD":
                ymax = 1.2
            plt.ylim(ymin, ymax)

            handles, labels = ax.get_legend_handles_labels()

            plt.tight_layout()
            fig.savefig(f"{OUTPUT_DIR}/{strat_type}_{stat}.png", dpi=300)
            plt.close(fig)

            fig_leg = plt.figure(figsize=(8, 1))
            fig_leg.legend(handles, labels, ncol=1, loc="center", frameon=False, fontsize=12)
            fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig_leg.savefig(f"{OUTPUT_DIR}/legend_{strat_type}.png",
                            dpi=300, bbox_inches="tight")
            plt.close(fig_leg)


def plot_together():

    for stat in STATS:
        fig, ax = plt.subplots(figsize=(3.6, 3.6))

        for strat_type in STRAT_TYPES:
            linestyle = "-" if strat_type == "splitwise" else "--"
            for strat_metric in STRAT_METRICS:
                if strat_metric in ["char_count", "FKGL"]:
                    means = [np.mean(score_tracker[stat][strat_type][strat_metric][sz]) for sz in SUBSET_SIZES]
                    stds  = [np.std (score_tracker[stat][strat_type][strat_metric][sz]) for sz in SUBSET_SIZES]
                    ax.plot(SUBSET_SIZES, means,
                            linestyle=linestyle,
                            label=f"{map_metric_name(strat_metric)} ({strat_type})",
                            color=METRIC_COLORS.get(strat_metric,"#000000"))
                    ax.fill_between(SUBSET_SIZES,
                                    np.array(means)-stds,
                                    np.array(means)+stds,
                                    alpha=0.2,
                                    color=METRIC_COLORS.get(strat_metric,"#000000"))

        ax.set_xlabel("subset size", fontsize=14)
        ax.set_ylabel(f"{stat} score", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # … your ylim + grid config …
        ax.grid(True, linestyle='--', alpha=0.7)

        all_means = [np.mean(score_tracker[stat][strat_type][strat_metric][size])
                    for strat_metric in STRAT_METRICS
                    for size in SUBSET_SIZES]
        ymin = max(min(all_means) - 0.005, 0)
        if stat == "KS":
            ymax = 0.04
        elif stat == "JSD":
            ymax = 0.06
        elif stat == "EMD":
            ymax = 1.1
        plt.ylim(ymin, ymax)

        # 1) Grab legend info but do NOT draw it here
        handles, labels = ax.get_legend_handles_labels()

        plt.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/both_{stat}.png", dpi=300)
        plt.close(fig)

        # 2) Draw & save legend only
        fig_leg = plt.figure(figsize=(8, 1))
        fig_leg.legend(handles, labels, ncol=1, loc="center", frameon=False, fontsize=12)
        fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_leg.savefig(f"{OUTPUT_DIR}/legend_both.png",
                        dpi=300, bbox_inches="tight")
        plt.close(fig_leg)


# === EXECUTE ===
plot_separately()
plot_together()
print(f"Plots saved to {OUTPUT_DIR}")
