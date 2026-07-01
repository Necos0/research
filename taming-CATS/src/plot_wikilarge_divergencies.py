import json
import os
import matplotlib.pyplot as plt

NUM_BINS = 15
DATASET_NAME = "wikilarge_ori"
EXPERIMENT_DIR = f"./../experiments/sample_from_{DATASET_NAME}/num_bins_{NUM_BINS}"
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(f"{EXPERIMENT_DIR}/all_divergence_results.json", "r") as f:
    DIVERGENCIES = json.load(f)

# Extract subset sizes and metric names
SUBSET_SIZES = sorted(map(int, DIVERGENCIES.keys()))  # Ensure sizes are in ascending order
STATS = ["JSD", "EMD", "KS"]
STRAT_TYPES = ["splitwise", "global"]

# Extract all stratification metrics (e.g., char_count, word_count, etc.)
ENTRY = next(iter(DIVERGENCIES.values()))
STRAT_METRICS = [entry[0] for entry in ENTRY["splitwise"]["JSD"]]
COLOR_MAP_FILE = "./../data/colormap/color_map.json"
with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)
METRIC_COLORS = COLOR_MAP["metrics"]

def plot_separately() -> None:

    for stat in STATS:

        for strat_type in STRAT_TYPES:

            plt.figure(figsize=(8, 5))
            
            for strat_metric in STRAT_METRICS:
                values = [
                    next(value for key, value in DIVERGENCIES[str(size)][strat_type][stat] if key == strat_metric)
                    for size in SUBSET_SIZES
                ]
                linestyle = "-" 
                plt.plot(
                        SUBSET_SIZES, 
                        values, 
                        linestyle=linestyle, 
                        color=METRIC_COLORS.get(strat_metric, "#000000"),
                        # marker=".", 
                        label=f"{strat_metric}"
                        )

            plt.xlabel("Subset Size")
            plt.ylabel(f"{stat} Score")
            plt.title(f"{stat} Divergence across Sampling Sizes ({strat_type})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/{strat_type}_{stat}.png", dpi=400)
            # plt.show()
            # plt.close()

def plot_together() -> None:

    for stat in STATS:

        plt.figure(figsize=(8, 5))

        for strat_type in STRAT_TYPES:
            
            for strat_metric in STRAT_METRICS:
                values = [
                    next(value for key, value in DIVERGENCIES[str(size)][strat_type][stat] if key == strat_metric)
                    for size in SUBSET_SIZES
                ]
                linestyle = "-" if strat_type == "splitwise" else "--"  # Solid for splitwise, dashed for global
                plt.plot(
                        SUBSET_SIZES, 
                        values, 
                        linestyle=linestyle, 
                        # marker=".", 
                        color=METRIC_COLORS.get(strat_metric, "#000000"),
                        label=f"{strat_metric} ({strat_type})"
                        )

        plt.xlabel("Subset Size")
        plt.ylabel(f"{stat} Score")
        plt.title(f"{stat} Divergence across Sampling Sizes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/both_{stat}.png", dpi=400)
        # plt.show()
        # plt.close()


# call both functions
plot_separately()
plot_together()

print(f"Plots saved to {OUTPUT_DIR}")

