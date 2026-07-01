"""
Load datasets directly from Hugging Face and generate distribution plots.
"""
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from helpers.hugging_face import load_dataset_from_hf

# Load environment variables from .env file
load_dotenv()


def load_colormap(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    

COLORMAP_PATH = "./data/colormap/color_map.json"
COLORMAP = load_colormap(COLORMAP_PATH)


def plot_distribution(dataset_name: str, metric_key: str, save_dir: str, combine_splits=True):
    """Load dataset from HF and plot distribution for a given metric."""
    
    # Load all splits and combine them
    if combine_splits:
        print(f"Loading and combining all splits for {dataset_name}...")
        train_data = load_dataset_from_hf(dataset_name, split="train", slice=-1)
        val_data = load_dataset_from_hf(dataset_name, split="validation", slice=-1)
        test_data = load_dataset_from_hf(dataset_name, split="test", slice=-1)
        
        # Combine all splits into one list
        data = list(train_data) + list(val_data) + list(test_data)
        print(f"Total samples: {len(data)} (train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)})")
    else:
        # Load only validation split
        data = load_dataset_from_hf(dataset_name, split="validation", slice=-1)
    
    source_vals = []
    target_vals = []
    
    for item in data:
        # Source and target metrics are lists containing dicts
        src_metrics = item.get("source_metrics", [{}])
        tgt_metrics = item.get("target_metrics", [{}])
        
        # Extract from first element if it's a list
        if isinstance(src_metrics, list) and len(src_metrics) > 0:
            src_metric = src_metrics[0].get(metric_key)
        else:
            src_metric = src_metrics.get(metric_key)
            
        if isinstance(tgt_metrics, list) and len(tgt_metrics) > 0:
            tgt_metric = tgt_metrics[0].get(metric_key)
        else:
            tgt_metric = tgt_metrics.get(metric_key)
        
        if src_metric is not None:
            source_vals.append(src_metric)
        if tgt_metric is not None:
            target_vals.append(tgt_metric)
    
    if not source_vals or not target_vals:
        print(f"No data for {metric_key} in {dataset_name}")
        return
    
    # Print statistics to check for unusual values
    print(f"\n{metric_key} statistics:")
    print(f"  Source - min: {min(source_vals):.2f}, max: {max(source_vals):.2f}, mean: {np.mean(source_vals):.2f}")
    print(f"  Target - min: {min(target_vals):.2f}, max: {max(target_vals):.2f}, mean: {np.mean(target_vals):.2f}")
    negative_count = sum(1 for v in source_vals + target_vals if v < 0)
    if negative_count > 0:
        print(f"  WARNING: {negative_count} negative values found!")
    
    # Determine bins - use fewer bins for cleaner plots
    all_vals = source_vals + target_vals
    bins = np.histogram_bin_edges(all_vals, bins=15)  # Adjust this number (default was 'auto')
    
    plt.figure(figsize=(4, 3))
    
    # Plot histograms
    n_source, bins_source, p_source = plt.hist(source_vals, bins=bins, alpha=0.7, label="Original", 
             color=COLORMAP["text_type"].get("source"), density=True, 
             edgecolor="black", histtype="stepfilled")
    n_target, bins_target, p_target = plt.hist(target_vals, bins=bins, alpha=0.7, label="Simplification", 
             color=COLORMAP["text_type"].get("target"), density=True, 
             edgecolor="black", histtype="stepfilled")
    
    # Calculate overlap
    source_hist, _ = np.histogram(source_vals, bins=bins, density=True)
    target_hist, _ = np.histogram(target_vals, bins=bins, density=True)
    overlap = np.minimum(source_hist, target_hist)
    
    p_overlap = plt.bar(bins[:-1], overlap, width=np.diff(bins), align="edge", 
            color=COLORMAP["overlap_color"], alpha=1, edgecolor="black", label="Overlap")
    
    # Add KDE curves
    kde_handles = []
    if len(set(source_vals)) > 1:
        line = sns.kdeplot(source_vals, color=COLORMAP["text_type"].get("source"), 
                   linewidth=1.5, label="Original KDE")
        kde_handles.append(line.get_lines()[-1])
    if len(set(target_vals)) > 1:
        line = sns.kdeplot(target_vals, color=COLORMAP["text_type"].get("target"), 
                   linewidth=1.5, label="Simplification KDE")
        kde_handles.append(line.get_lines()[-1])
    
    plt.xlabel(f'{metric_key}', fontsize=18)
    plt.ylabel('')  # Remove y-axis label
    plt.tick_params(axis='both', labelsize=16)
    
    # Force x-axis ticks to be integers with fewer ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    
    # Format x-axis labels: use "k" notation for values >= 1000
    def format_thousands(x, pos):
        if x >= 1000:
            val = x / 1000
            if val == int(val):
                return f'{int(val)}k'
            else:
                return f'{val:.1f}k'
        return f'{int(x)}'
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{metric_key}_hist.png", dpi=400, bbox_inches='tight')
    
    # Save separate legend
    fig_leg = plt.figure(figsize=(6, 1))
    handles = [p_source[0], p_target[0], p_overlap[0]] + kde_handles
    labels = ["Original", "Simplification", "Overlap", "Original KDE", "Simplification KDE"]
    labels = labels[:len(handles)]  # Only use as many labels as we have handles
    fig_leg.legend(handles, labels, loc='center', ncol=3, frameon=False, fontsize=16)
    fig_leg.tight_layout()
    fig_leg.savefig(f"{save_dir}/{metric_key}_hist_legend.png", dpi=400, bbox_inches='tight')
    plt.close('all')
    
    print(f"Saved {metric_key} distribution plot to {save_dir}")


def main():
    # Define your HF dataset name
    DATASET_NAME = "Med-EASi"  # Change this to your dataset
    SAVE_DIR = "./data/datasets/med-easi/stats/distributions"
    
    # Metrics to plot - use exact names from the data
    metrics = ["FKGL", "ARI", "Dale-Chall", "FRE", "char_count", "word_count"]
    
    for metric in metrics:
        try:
            plot_distribution(DATASET_NAME, metric, SAVE_DIR)
        except Exception as e:
            print(f"Error plotting {metric}: {e}")


if __name__ == "__main__":
    main()
