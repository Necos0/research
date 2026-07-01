import json
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde


COLOR_MAP_FILE = "./../data/colormap/color_map.json"
with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)

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

def set_random_seeds(num_seeds=10):
    random.seed(42)
    SEEDS = random.sample(range(0, 2**32 - 1), num_seeds)
    return SEEDS

def load_jsonl(filepath):
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_metrics(data, metrics):
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def remove_outliers_by_char_length(data: list, lower_percentile=3, upper_percentile=97) -> list:
    """Remove entries based on character length outliers (3rd and 97th percentiles)."""
    
    # Calculate the character length for each entry
    char_lengths = [len(line["source_text"]) for line in data]

    # Calculate the lower and upper percentiles for character length
    lower_threshold = np.percentile(char_lengths, lower_percentile)
    upper_threshold = np.percentile(char_lengths, upper_percentile)

    # Filter out entries based on the character length thresholds
    filtered_data = [
        line for line in data
        if lower_threshold <= len(line["source_text"]) <= upper_threshold
    ]

    return filtered_data

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

def generate_splits(data, metric_values, strat_metric, num_bins=35, seed=None):
    """Generate stratified train, validation, and test splits."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})

    rng = np.random.default_rng(seed) # use local random generator
    
    train_data, val_data, test_data = [], [], []

    for bin_id in np.unique(bin_indices):
        bin_data = df[df["bin"] == bin_id]["data"].tolist()
        rng.shuffle(bin_data)  # Use the local random generator to shuffle
        train_data.extend(bin_data[:int(len(bin_data) * 0.8)])  # 80% train
        val_data.extend(bin_data[int(len(bin_data) * 0.8):int(len(bin_data) * 0.9)])  # 10% validation
        test_data.extend(bin_data[int(len(bin_data) * 0.9):])  # 10% test

    # Return non-overlapping splits
    return train_data, val_data, test_data

def plot_distributions_on_one_image_shadow(metrics,
                                    full_vals_seeds,   # dict: seed → metric → values
                                    train_vals_seeds,
                                    val_vals_seeds,
                                    test_vals_seeds,
                                    save_dir,
                                    dataset_name=None,
                                    strat_metric=None,
                                    nbins=25
                                    ):
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)

    num_metrics = len(metrics)
    cols = 5
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    axes = axes.flatten()

    # Precompute a common x‐grid for each metric
    grid_dict = {}
    for metric in metrics:
        all_data = []
        for d in (full_vals_seeds, train_vals_seeds, val_vals_seeds, test_vals_seeds):
            for seed in d:
                all_data += d[seed][metric]
        low, high = np.percentile(all_data, [0, 100])
        grid_dict[metric] = np.linspace(low, high, 100)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        xs = grid_dict[metric]

        def plot_shadow(vals_seeds, color, label):
            # build densities matrix: (n_seeds × len(xs))
            dens = np.stack([
                gaussian_kde(vals_seeds[seed][metric])(xs)
                for seed in vals_seeds
            ])
            mean = dens.mean(axis=0)
            std  = dens.std(axis=0)

            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=0.2)
            ax.plot(xs, mean, color=color, lw=1.5, label=label)

        # draw full, train, val, test with shadows:
        plot_shadow(full_vals_seeds,  COLOR_MAP["splits"]["full"],  "Full")
        plot_shadow(train_vals_seeds, COLOR_MAP["splits"]["train"], "Train")
        plot_shadow(val_vals_seeds,   COLOR_MAP["splits"]["val"],   "Validation")
        plot_shadow(test_vals_seeds,  COLOR_MAP["splits"]["test"],  "Test")

        ax.set_xlabel(map_metric_name(metric), fontsize=16)
        if i % cols == 0:
            ax.set_ylabel("density", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.tick_params(labelsize=16, rotation=45)
        ax.tick_params(
            axis='x',
            labelsize=16,
            labelrotation=45
        )
        ax.tick_params(
            axis='y',
            labelsize=16,
            labelrotation=0
        )

        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig_path = f"{save_dir}/all_metrics_shadow_{nbins}_{map_metric_name(strat_metric)}_{dataset_name}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    # legend-only
    fig_leg = plt.figure(figsize=(10, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig_leg.legend(handles, labels, ncol=5, loc="center", frameon=False, fontsize=12)
    fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
    leg_path = f"{save_dir}/all_metrics_legend.png"
    fig_leg.savefig(leg_path, dpi=300, bbox_inches="tight")
    plt.close(fig_leg)

def plot_distributions_on_one_image_shadow_counts(metrics,
                                    full_vals_seeds,
                                    train_vals_seeds,
                                    val_vals_seeds,
                                    test_vals_seeds,
                                    save_dir,
                                    dataset_name=None,
                                    strat_metric=None,
                                    nbins=25
                                    ):
    save_dir = f"{save_dir}"
    os.makedirs(save_dir, exist_ok=True)

    num_metrics = len(metrics)
    cols = 5
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
    axes = axes.flatten()

    # 1) Precompute bin-edges for each metric
    bins_dict = {}
    for metric in metrics:
        all_data = []
        for d in (full_vals_seeds, train_vals_seeds, val_vals_seeds, test_vals_seeds):
            for seed in d:
                all_data += d[seed][metric]
        low, high = np.percentile(all_data, [0, 100])
        bins = np.linspace(low, high, nbins)  
        # bins = np.histogram_bin_edges(all_data, bins='fd')  # Freedman–Diaconis rule
        # centers = (bins[:-1] + bins[1:]) / 2
        bins_dict[metric] = bins

    for i, metric in enumerate(metrics):
        ax = axes[i]
        bins = bins_dict[metric]
        centers = (bins[:-1] + bins[1:]) / 2

        def plot_shadow_counts(vals_seeds, color, label):
            # compute histogram for each seed
            counts = np.stack([
                np.histogram(vals_seeds[seed][metric], bins=bins)[0]
                for seed in vals_seeds
            ])  # shape = (n_seeds, n_bins)
            mean = counts.mean(axis=0)
            std  = counts.std(axis=0)

            ax.fill_between(centers, mean - std, mean + std,
                            color=color, alpha=0.2)
            ax.plot(centers, mean, color=color, lw=1.5, label=label)

        # 2) plot each split’s shadowed counts
        plot_shadow_counts(full_vals_seeds,  COLOR_MAP["splits"]["full"],  "Full")
        plot_shadow_counts(train_vals_seeds, COLOR_MAP["splits"]["train"], "Train")
        plot_shadow_counts(val_vals_seeds,   COLOR_MAP["splits"]["val"],   "Validation")
        plot_shadow_counts(test_vals_seeds,  COLOR_MAP["splits"]["test"],  "Test")

        ax.set_xlabel(map_metric_name(metric), fontsize=16)
        if i % cols == 0:
            ax.set_ylabel("count", fontsize=16)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis='x', labelsize=16, rotation=45)
        ax.tick_params(axis='y', labelsize=16)
        ax.grid(True, linestyle="--", alpha=0.5)

        # more ticks if you like
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune=None))

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    fig_path = f"{save_dir}/all_metrics_shadow_counts_{nbins}_{map_metric_name(strat_metric)}_{dataset_name}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # legend-only
    fig_leg = plt.figure(figsize=(8, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig_leg.legend(handles, labels, ncol=5, loc="center", frameon=False, fontsize=12)
    fig_leg.subplots_adjust(left=0, right=1, top=1, bottom=0)
    leg_path = f"{save_dir}/all_metrics_legend.png"
    fig_leg.savefig(leg_path, dpi=300, bbox_inches="tight")
    plt.close(fig_leg)

def main():
    DATASETS = [
        "medeasi",
        "newsela",
        "simpa",
        # "wikilarge_ori"
        ]
    DATA_DIR = "./../data"
    BINS = [
            15,
            25,
            35,
            45
            ]
    METRICS = [
                "char_count", 
                "word_count", 
                "FKGL", 
                "ARI", 
                "Dale-Chall"
                ]

    seeds = set_random_seeds(num_seeds=10)
    
    EXPERIMENT_RESULTS = []

    for NUM_BINS in BINS:
        for dataset_name in DATASETS:
            print(f"Processing {dataset_name}...")
            dataset_path = f"{DATA_DIR}/datasets/{dataset_name}/dataset.jsonl"
            for STRAT_METRIC in METRICS:
                print(f"Stratifying by {STRAT_METRIC}...")
                full_vals_seeds  = {}
                train_vals_seeds = {}
                val_vals_seeds   = {}
                test_vals_seeds  = {}
                for seed in seeds:
                    print(f"\n--- Using seed: {seed} ---")
                    
                    SAVE_DIR = f"./../experiments/splits_sampling_w_outliers/stratified_by_{STRAT_METRIC}"
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    
                    data = load_jsonl(dataset_path)
                    # data = remove_outliers_by_char_length(data, lower_percentile=3, upper_percentile=97)
                    
                    train_data, val_data, test_data = generate_splits(data, extract_metrics(data, METRICS), STRAT_METRIC, num_bins=NUM_BINS, seed=seed)
                
                    full_metric_values = extract_metrics(data, METRICS)
                    train_metric_values = extract_metrics(train_data, METRICS)
                    val_metric_values = extract_metrics(val_data, METRICS)
                    test_metric_values = extract_metrics(test_data, METRICS)

                    full_vals_seeds[seed]  = extract_metrics(data,   METRICS)
                    train_vals_seeds[seed] = extract_metrics(train_data, METRICS)
                    val_vals_seeds[seed]   = extract_metrics(val_data,   METRICS)
                    test_vals_seeds[seed]  = extract_metrics(test_data,  METRICS)

                    # Calculate KL Divergence between distributions
                    ks_train,_ = ks_2samp(full_metric_values[STRAT_METRIC], train_metric_values[STRAT_METRIC])
                    ks_val,_ = ks_2samp(full_metric_values[STRAT_METRIC], val_metric_values[STRAT_METRIC])
                    ks_test,_ = ks_2samp(full_metric_values[STRAT_METRIC], test_metric_values[STRAT_METRIC])

                    # Store results for each experiment
                    EXPERIMENT_RESULTS.append({
                        "seed": int(seed),
                        "dataset": dataset_name,
                        "strat_metric": STRAT_METRIC,
                        "num_bins": NUM_BINS,
                        "KS_full_train": ks_train,
                        "KS_full_val": ks_val,
                        "KS_full_test": ks_test,
                        "average_KS": np.mean([ks_train, ks_val, ks_test])
                    })

                plot_distributions_on_one_image_shadow(
                    METRICS,
                    full_vals_seeds,
                    train_vals_seeds,
                    val_vals_seeds,
                    test_vals_seeds,
                    SAVE_DIR,
                    dataset_name=dataset_name,
                    strat_metric=STRAT_METRIC,
                    nbins=NUM_BINS
                )
                plot_distributions_on_one_image_shadow_counts(
                    METRICS,
                    full_vals_seeds,
                    train_vals_seeds,
                    val_vals_seeds,
                    test_vals_seeds,
                    SAVE_DIR,
                    dataset_name=dataset_name,
                    strat_metric=STRAT_METRIC,
                    nbins=NUM_BINS
                )

    with open(f"./../experiments/splits_sampling/all_results.json", "w") as f:
        json.dump(EXPERIMENT_RESULTS, f, indent=4)
        
if __name__ == "__main__":
    main()


