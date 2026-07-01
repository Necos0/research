import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr


def load_colormap(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
COLORMAP_PATH = "./data/colormap/color_map.json"
COLORMAP = load_colormap(COLORMAP_PATH)

def load_jsonl(filepath: str) -> list:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_correlation_strength(corr_coeff: float) -> str:
    """Return the correlation strength description based on the correlation coefficient."""
    if abs(corr_coeff) < 0.2:
        return "Very weak"
    elif abs(corr_coeff) < 0.4:
        return "Weak"
    elif abs(corr_coeff) < 0.6:
        return "Moderate"
    elif abs(corr_coeff) < 0.8:
        return "Strong"
    else:
        return "Very Strong"
    
def get_significance_level(p_value: float) -> str:
    """Return significance level based on the p-value."""
    if p_value <= 0.001:
        return "***"  # Very highly significant
    elif p_value <= 0.01:
        return "**"  # Highly significant
    elif p_value <= 0.05:
        return "*"  # Significant
    else:
        return "n.s."  # Not significant

def plot_compression(data: list, metric: str, dataset_dir: str) -> None:
    source_lengths, target_lengths = get_compression_values(data, metric)

    # Check if either source_lengths or target_lengths is empty
    if not source_lengths or not target_lengths:
        print(f"Warning: No valid values for metric '{metric}' found. Skipping plot for this metric.")
        return  # Skip plotting for this metric if no values are found

    # Plotting the distribution shift
    min_val = min(min(source_lengths), min(target_lengths))
    max_val = max(max(source_lengths), max(target_lengths))
    bins = np.linspace(min_val, max_val, 25)

    # Plotting the distribution shift
    
    ############ histograms with aligned bins ############
    # plt.subplot(1, 2, 1)

    plt.figure(figsize=(5, 5))

    # Plot source and target histograms separately with specific alpha values for transparency
    n_source, bins_source, _ = plt.hist(source_lengths, bins=bins, alpha=0.7, label="Original", color=COLORMAP["text_type"].get("source"), density=True, edgecolor="black", histtype="stepfilled")
    n_target, bins_target, _ = plt.hist(target_lengths, bins=bins, alpha=0.7, label="Simplification", color=COLORMAP["text_type"].get("target"), density=True, edgecolor="black", histtype="stepfilled")

    # Compute and plot overlap
    # Calculate histogram values for source and target using the same bins
    source_hist, _ = np.histogram(source_lengths, bins=bins, density=True)
    target_hist, _ = np.histogram(target_lengths, bins=bins, density=True)

    # Calculate overlap using minimum of the source and target histograms
    overlap = np.minimum(source_hist, target_hist)

    # Plot the overlap as a bar plot with gray color
    # We use the bins[:-1] for proper alignment and width as np.diff(bins)
    plt.bar(bins[:-1], overlap, width=np.diff(bins), align="edge", color=COLORMAP["overlap_color"], alpha=1, edgecolor="black", label="Overlap")

    # Add KDE for smooth distribution curves
    if len(set(source_lengths)) > 1:
        sns.kdeplot(source_lengths, color=COLORMAP["text_type"].get("source"), linewidth=1.5, label="Original KDE")
    if len(set(target_lengths)) > 1:
        sns.kdeplot(target_lengths, color=COLORMAP["text_type"].get("target"), linewidth=1.5, label="Simplification KDE")

    if metric in ["char_count", "sentence_count", "word_count"]:
        plt.xlabel(f'{metric} Count', fontsize=18)
    else:
        plt.xlabel(f'{metric}', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=14)
    output_dir = f"{dataset_dir}/stats/distributions"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_hist.png", dpi=400, bbox_inches='tight')

    ############ scatter plot ############
    plt.figure(figsize=(5, 5))

    plt.scatter(
        target_lengths, 
        source_lengths, 
        edgecolor=COLORMAP["common_color"],  # Edge color of the dots
        linewidth=0.5,
        alpha=0.4, 
        color=COLORMAP["common_color"], 
        label="Source vs Target"
        )

    if len(source_lengths) > 1 and len(target_lengths) > 1:
        corr_coeff, p_value = pearsonr(source_lengths, target_lengths)

        # compute correlation strength
        correlation_strength = get_correlation_strength(corr_coeff)
        # compute significance level
        p_val_interpretation = get_significance_level(p_value)

        avg_source = np.mean(source_lengths)
        avg_target = np.mean(target_lengths)
        slope = avg_source / avg_target if avg_target != 0 else 0

        line_x = np.linspace(0, max(target_lengths), 100)
        line_y = slope * line_x
        plt.plot(line_x, line_y, color=COLORMAP["common_color"], linestyle='-', label="Mean value")

        plt.text(0.05, 0.95, f"Pearson: {corr_coeff:.4f} ({correlation_strength})", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', color='black')

    # Set the same scale for both axes
    min_val = 0
    max_val = max(max(source_lengths), max(target_lengths))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(f'Simplification {metric} Count')
    plt.ylabel(f'Original {metric} Count')
    plt.title(p_val_interpretation)

    # Save the plot
    output_dir = f"{dataset_dir}/stats/distributions"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_scatter.png", dpi=400)

def get_compression_values(data: list, metric: str) -> tuple:
    source_val_arr = []
    target_val_arr = []

    for line in data:

        simplifications = line["simplifications"]
        for simplification in simplifications:
            if metric in ["char", "sentence", "word"]:
                target_value = simplification["target_metrics"][f"{metric}_count"]
                target_val_arr.append(target_value)
                source_value = line["source_metrics"][f"{metric}_count"]
                source_val_arr.append(source_value)
            else:
                target_value = simplification["target_metrics"][f"{metric}"]
                target_val_arr.append(target_value)
                source_value = line["source_metrics"][f"{metric}"]
                source_val_arr.append(source_value)
    
    return source_val_arr, target_val_arr

def get_eval_values(data: list, metric: str) -> list:
    target_val_arr = []

    for line in data:
        simplifications = line["simplifications"]
        for simplification in simplifications:
            if metric in ["BLEU", "BERTScore"]:
                target_value = simplification["target_metrics"][f"{metric}"]
                target_val_arr.append(target_value)
    
    return target_val_arr

def plot_eval_values(data: list, metric: str, dataset_dir: str) -> None:
    """ Plot BLEU and BERTScore. """

    target_vals = get_eval_values(data, metric)

    # Plotting the distribution shift
    # Calculate the common bin edges for source and target distributions
    min_val = min(target_vals)
    max_val = max(target_vals)
    bins = np.linspace(min_val, max_val, 25)  # TODO adjust n bins to ompimize the visual effect 

    ############ histograms with aligned bins ############
    plt.figure(figsize=(5, 5))
    plt.hist(target_vals, bins=bins, alpha=0.7, label="Simplification", color=COLORMAP["common_color"], density=True, edgecolor="black")
    # Add KDE for smooth distribution curves
    if len(set(target_vals)) > 1:
        sns.kdeplot(target_vals, color=COLORMAP["common_color"], linewidth=1.5, label="Simplification KDE")
    plt.xlabel(f'{metric}')
    plt.ylabel('Frequency')
    plt.legend()
    output_dir = f"{dataset_dir}/stats/distributions"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_hist.png", dpi=400)

    ############ violin plot ############
    plt.figure(figsize=(4, 3))
    sns.violinplot(x=target_vals, color=COLORMAP["common_color"], alpha=0.7, inner="quartile")
    sns.boxplot(x=target_vals, color='black', width=0.15, fliersize=3)  # Add box plot for summary stats
    # plt.xlabel(f'{metric} Score')
    # plt.title(f'Distribution of {metric} Scores')
    
    # save plots
    output_dir = f"{dataset_dir}/stats/distributions"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_violin.png", dpi=300)
    
def save_log(data: list, dataset_name: str, dataset_dir: str, comparison_metrics: list, similarity_metrics: list) -> None:
    log_data = {}
    
    for metric in comparison_metrics:
        source_vals, target_vals = get_compression_values(data, metric)
        if source_vals and target_vals:
            log_data[metric] = {
                "source_mean": np.mean(source_vals),
                "source_std": np.std(source_vals),
                "source_median": np.median(source_vals),
                "target_mean": np.mean(target_vals),
                "target_std": np.std(target_vals),
                "target_median": np.median(target_vals)
            }
    
    if "wikilarge" not in dataset_name:
        for metric in similarity_metrics:
            target_vals = get_eval_values(data, metric)
            if target_vals:
                log_data[metric] = {
                    "mean": np.mean(target_vals),
                    "std": np.std(target_vals),
                    "median": np.median(target_vals)
                }
    
    log_path = os.path.join(dataset_dir, "stats", "dataset_stats.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"Saved log for {dataset_name} at {log_path}")

def plot_source_target_comparison(source_vals: list, target_vals: list, metric: str, dataset_dir: str) -> None:

    # Threshold values at 0 (set negative values to 0)
    source_vals = np.maximum(source_vals, 0)
    target_vals = np.maximum(target_vals, 0)

    df = pd.DataFrame({"Source": source_vals, "Target": target_vals})

    metric_color = COLORMAP["metrics"].get(metric)
    source_color = COLORMAP["text_type"].get("source")
    target_color = COLORMAP["text_type"].get("target")

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df, palette=[source_color, target_color], width=0.3)
    
    # Plot trajectory from median to median and mean to mean
    source_mean, target_mean = np.mean(source_vals), np.mean(target_vals)
    source_median, target_median = np.median(source_vals), np.median(target_vals)
    
    plt.plot([0, 1], [source_mean, target_mean], color="black", marker="o", linestyle="solid",label="Mean")
    plt.plot([0, 1], [source_median, target_median], color="black", marker="o", linestyle="dashed", label="Median")
    
    plt.xticks([0, 1], ["Original", "Simplification"])
    plt.ylabel(metric)
    # plt.title(f"Comparison of Source and Target for {metric}")
    plt.legend()
    plt.grid()
    
    output_dir = f"{dataset_dir}/stats/trajectory"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    plt.close()

def plot_qq_plot(data: list, metric: str, dataset_dir: str) -> None:
    """
    Create a QQ plot for the given metric.
    
    """

    plt.figure(figsize=(4, 4))

    source_vals, target_vals = get_compression_values(data, metric)

    # Threshold values at 0 (set negative values to 0)
    source_vals = np.maximum(source_vals, 0)
    target_vals = np.maximum(target_vals, 0)

    # Sort values
    source_vals = np.sort(source_vals)
    target_vals = np.sort(target_vals)

    percentiles = np.linspace(5, 95, 100) # exclude the extreme values
    source_quantiles = np.percentile(source_vals, percentiles)
    target_quantiles = np.percentile(target_vals, percentiles)

    # plot the quantiles as a scatter plot
    plt.scatter(y=source_vals, x=target_vals, color=COLORMAP["common_color"], alpha=0.7)

    # fit the line taking into account the percentiles
    z = np.polyfit(y=source_quantiles, x=target_quantiles, deg=1)
    p = np.poly1d(z)
    plt.plot(source_quantiles, p(source_quantiles), color="black", linestyle="--", label="Fit Line (5th-95th Percentiles)")

    plt.xlabel(f'Simplification {metric}')
    plt.ylabel(f'Original {metric}')
    plt.legend()
    # plt.title(f'QQ Plot for {metric}')
    plt.grid()
    plt.tight_layout()    

    output_dir = f"{dataset_dir}/stats/qq_plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{metric}.png", dpi=400)
    plt.close()

def main():

    DATASETS = [
        "simpa",
        # "simpa_lexical",
        # "simpa_syntactic",
        # "newsela",
        # "medeasi",
        # "wikilarge_ori_global",
        # "wikilarge_ori_splitwise",
        # "wikilarge_global",
        # "wikilarge_splitwise",
        # "wikilarge"
    ]
    DATA_DIR = "./data" 

    for DATASET in DATASETS:
        DATASET_DIR = f"{DATA_DIR}/datasets/{DATASET}"
        DATASET_PATH = f"{DATASET_DIR}/dataset.jsonl"

        dataset = load_jsonl(DATASET_PATH)

        comparison_metrics = ["char", "word", "FRE", "ARI", "FKGL", "Dale-Chall"]
        similarity_metrics = ["BLEU", "BERTScore"]
        
        for metric in comparison_metrics:
            plot_qq_plot(data=dataset, metric=metric, dataset_dir=DATASET_DIR)
            plot_compression(data=dataset, metric=metric, dataset_dir=DATASET_DIR)
            source_vals, target_vals = get_compression_values(dataset, metric)
            plot_source_target_comparison(source_vals, target_vals, metric, DATASET_DIR)
        
        if DATASET != "wikilarge":
            for metric in similarity_metrics:
                plot_eval_values(data=dataset, metric=metric, dataset_dir=DATASET_DIR)

        save_log(data=dataset, dataset_name=DATASET, dataset_dir=DATASET_DIR, 
                 comparison_metrics=comparison_metrics, similarity_metrics=similarity_metrics)


if __name__ == "__main__":
    main()