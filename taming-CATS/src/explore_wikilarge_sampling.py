import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from classes.Metrics import Metrics

def load_jsonl(filepath: str) -> list:
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: list, filepath: str) -> None:
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            source_text = line["source_text"]
            for simplification in line["simplifications"]:
                bleu = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bleu()
                bertscore = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bertscore()
                
                # Append the scores to the target_metrics
                simplification['target_metrics']['BLEU'] = bleu
                simplification['target_metrics']['BERTScore'] = bertscore

            f.write(json.dumps(line) + "\n")

def remove_outliers_by_char_length(data: list, lower_percentile=5, upper_percentile=95) -> list:
    """Remove entries based on character length outliers."""
    
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

def remove_outliers_by_fkgl(data, lower_percentile=5, upper_percentile=95) -> list:
    """Remove entries based on FKGL outliers."""
    
    # Calculate the FKGL for each entry
    fkgl_values = [line["source_metrics"].get("FKGL", 0) for line in data]

    # Calculate the lower and upper percentiles for FKGL
    lower_threshold = np.percentile(fkgl_values, lower_percentile)
    upper_threshold = np.percentile(fkgl_values, upper_percentile)

    # Filter out entries based on the FKGL thresholds
    filtered_data = [
        line for line in data
        if lower_threshold <= line["source_metrics"].get("FKGL", 0) <= upper_threshold
    ]

    return filtered_data

def filter_by_fkgl(data, min_threshold=0.0):
    filtered_data = [
        entry for entry in data
        if entry["source_metrics"].get("FKGL", float('inf')) >= min_threshold
        and entry["simplifications"][0]["target_metrics"].get("FKGL", float('inf')) >= min_threshold
    ]
    return filtered_data

def filter_by_ari(data, min_threshold=0.0):
    filtered_data = [
        entry for entry in data
        if entry["source_metrics"].get("ARI", float('inf')) >= min_threshold
        and entry["simplifications"][0]["target_metrics"].get("ARI", float('inf')) >= min_threshold
    ]
    return filtered_data

def extract_metrics(data: list, metrics: list) -> dict:
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def stratified_sampling(data: list, metric_values: dict, metric: str, num_bins=20, subset_size=2000, seed=42) -> list:
    """Perform stratified sampling based on a single metric."""
    values = np.array(metric_values[metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=subset_size / len(df), random_state=seed))
    return subset["data"].tolist()

def stratified_sampling_from_split(data: list, metric_values: dict, metric: str, num_bins=20, subset_size=2000, seed=42):
    """Perform stratified sampling separately for train, valid, and test to maintain proportions."""
    # Split dataset into train, valid, test
    train_data = [line for line in data if line["metadata"]["original_split"] == "train"]
    valid_data = [line for line in data if line["metadata"]["original_split"] == "valid"]
    test_data = [line for line in data if line["metadata"]["original_split"] == "test"]

    train_size = int(subset_size * 0.8)
    valid_size = int(subset_size * 0.1)
    test_size = int(subset_size * 0.1)

    train_metrics = extract_metrics(train_data, [metric])
    valid_metrics = extract_metrics(valid_data, [metric])
    test_metrics = extract_metrics(test_data, [metric])

    def sample_split(split_data: list, split_metrics: dict, split_size: int) -> list:
        """Helper function to perform stratified sampling for each split."""
        if len(split_data) == 0:
            return []  # In case a split has no data (edge case)
        values = np.array(split_metrics[metric])
        bins = np.histogram_bin_edges(values, bins=num_bins)
        bin_indices = np.digitize(values, bins)
        df = pd.DataFrame({"data": split_data, "bin": bin_indices})
        subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=split_size / len(df), random_state=seed))
        return subset["data"].tolist()

    # Sample from each split
    sampled_train = sample_split(train_data, train_metrics, train_size)
    sampled_valid = sample_split(valid_data, valid_metrics, valid_size)
    sampled_test = sample_split(test_data, test_metrics, test_size)

    return sampled_train + sampled_valid + sampled_test

def plot_final_multi_metric(metric_values: dict, all_subset_metric_values: dict, strat_metrics: list, metrics: list, subset_dir: str) -> None:
    """Create a single multi-plot where each row represents a stratification metric."""
    num_strat_metrics = len(strat_metrics)
    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_strat_metrics, num_metrics, figsize=(4 * num_metrics, 4 * num_strat_metrics), constrained_layout=True)

    for row, strat_metric in enumerate(strat_metrics):
        subset_metric_values = all_subset_metric_values[strat_metric]
        
        axes[row, 0].annotate(
            f"Stratified by {strat_metric}",
            xy=(0, 0.5),
            xytext=(-axes[row, 0].yaxis.labelpad - 30, 0),
            xycoords=axes[row, 0].yaxis.label,
            textcoords="offset points",
            size=12,
            ha="right",
            va="center",
            rotation=90,
            fontweight="bold"
        )
        
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            sns.kdeplot(metric_values[metric], color="blue", label="Full Dataset", ax=ax)
            sns.kdeplot(subset_metric_values[metric], color="red", label="Subset", ax=ax)
            ax.set_title(f"{metric}")
            
            if col == 0:
                ax.set_ylabel("Density")
            if row == num_strat_metrics - 1:
                ax.set_xlabel(metric)
            if row == 0 and col == num_metrics - 1:
                ax.legend()

    plot_filepath = f"{subset_dir}/stratification_metrics.png"
    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    plt.savefig(plot_filepath, dpi=400)
    plt.close()

def compute_similarity_scores(full_data: dict, subset_data: dict, metrics: list, n_bins=25) -> dict:
    """ Compute JSD, EMD, and KS scores to quantify similarity between distributions.

        JSD: Jensen–Shannon divergence
        EMD: Earth Mover's Distance
        KS: Kolmogorov-Smirnov Statistic
    
    """
    scores = {metric: {"JSD": None, "EMD": None, "KS": None} for metric in metrics}
    
    for metric in metrics:
        full_dist = np.array(full_data[metric])
        subset_dist = np.array(subset_data[metric])

        bins = np.histogram_bin_edges(np.concatenate([full_dist, subset_dist]), bins=n_bins)
        full_hist, _ = np.histogram(full_dist, bins=bins, density=True)
        subset_hist, _ = np.histogram(subset_dist, bins=bins, density=True)
        jsd = jensenshannon(full_hist, subset_hist)

        emd = wasserstein_distance(full_dist, subset_dist)

        ks_stat, _ = ks_2samp(full_dist, subset_dist)

        scores[metric] = {"JSD": jsd, "EMD": emd, "KS": ks_stat}

    return scores

def rank_stratifications(all_scores: dict) -> dict:
    """Compute separate rankings for JSD, EMD, and KS scores across all stratification methods."""
    strat_ranking = {metric: {} for metric in ["JSD", "EMD", "KS"]}

    for strat_metric, metric_scores in all_scores.items():
        strat_ranking["JSD"][strat_metric] = np.mean([metric_scores[m]["JSD"] for m in metric_scores])
        strat_ranking["EMD"][strat_metric] = np.mean([metric_scores[m]["EMD"] for m in metric_scores])
        strat_ranking["KS"][strat_metric] = np.mean([metric_scores[m]["KS"] for m in metric_scores])

    # Rank each metric separately (lower scores mean more similarity)
    ranked_strats = {
        metric: sorted(strat_ranking[metric].items(), key=lambda x: x[1])
        for metric in ["JSD", "EMD", "KS"]
    }

    return ranked_strats

# def save_ranking_log(ranked_strats: dict, subset_dir: str) -> None:
#     """Save stratification ranking to a log file and a JSON file."""
#     log_filepath = f"{subset_dir}/metric_rank_log.txt"
#     json_filepath = f"{subset_dir}/metric_rank_log.json"
    
#     os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

#     with open(log_filepath, "w", encoding="utf-8") as f:
#         f.write("=== Stratification Rankings (Lower Score = More Representative) ===\n\n")
#         for metric, rankings in ranked_strats.items():
#             f.write(f"--- {metric} Ranking ---\n")
#             for rank, (strat, score) in enumerate(rankings, start=1):
#                 f.write(f"{rank}. Stratified by {strat}: {score:.4f}\n")
#             f.write("\n")

#     with open(json_filepath, "w", encoding="utf-8") as f:
#         json.dump(ranked_strats, f, indent=4)

def save_json(data: dict, filepath: str) -> None:
    """Save dictionary data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def main():

    dataset_name = "wikilarge_ori"
    experiment = "explore_sampling"
    stratification_types = [
        "splitwise", 
        "global"
        ] 
    full_dataset_path = f"./../data/datasets/{dataset_name}/dataset.jsonl"
    experiment_dir = f"./../experiments/sample_from_{dataset_name}"
    log_output_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(log_output_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    json_output_path = f"{experiment_dir}/all_divergence_results.json"

    seeds = [69, 1, 40, 7, 29, 48, 78, 34, 67, 84]
    bin_values = [15, 25, 35, 45]
    metrics = ["char_count", "word_count", "FKGL", "ARI", "Dale-Chall"]
    
    data = load_jsonl(full_dataset_path)
    data = remove_outliers_by_char_length(data) # remove outliers by char length
    data = remove_outliers_by_fkgl(data) # remove outliers by FKGL
    data = filter_by_fkgl(data) # remove outliers by FKGL - with thresholding
    data = filter_by_ari(data) # remove outliers by ARI - with thresholding
    metric_values = extract_metrics(data, metrics)
    
    if experiment  == "explore_sampling":
        if os.path.exists(json_output_path):
            with open(json_output_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        else:
            all_results = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        if seed_key not in all_results:
            all_results[seed_key] = {}

        for num_bins in bin_values:
            bins_key = f"num_bins_{num_bins}"
            if bins_key not in all_results[seed_key]:
                all_results[seed_key][bins_key] = {}

            for subset_size in range(50, 3000, 50):
                subset_key = f"subset_size_{subset_size}"
                if subset_key in all_results[seed_key][bins_key]:
                    print(f"Skipping seed={seed}, bins={num_bins}, subset={subset_size}")
                    continue
                all_results[seed_key][bins_key][subset_key] = {}

                for strat_type in stratification_types:
                    all_subset_metric_values = {}
                    all_similarity_scores = {}

                    sampling_fn = stratified_sampling_from_split if strat_type == "splitwise" else stratified_sampling

                    for strat_metric in metrics:
                        subset_data = sampling_fn(
                            data=data,
                            metric_values=metric_values,
                            metric=strat_metric,
                            num_bins=num_bins,
                            subset_size=subset_size,
                            seed=seed
                        )

                        subset_metric_values = extract_metrics(subset_data, metrics)
                        all_subset_metric_values[strat_metric] = subset_metric_values
                        all_similarity_scores[strat_metric] = compute_similarity_scores(metric_values, subset_metric_values, metrics)

                    ranked_strats = rank_stratifications(all_similarity_scores)
                    all_results[seed_key][bins_key][subset_key][strat_type] = ranked_strats

                save_json(all_results, json_output_path)
                print(f"Saved seed={seed}, bins={num_bins}, subset={subset_size}")

    # elif experiment == "create_subset":

    #     subset_size = 2000
    #     num_bins = num_bins
    #     strat_metric = "FKGL"

    #     # Run for both stratification types
    #     for stratification_type in stratification_types:
    #         output_dir = f"./../data/datasets/{dataset_name}_{stratification_type}_{subset_size}"
    #         os.makedirs(output_dir, exist_ok=True)

    #         all_subset_metric_values = {}
    #         all_similarity_scores = {}

    #         if stratification_type == "splitwise":
    #             sampling_function = stratified_sampling_from_split
    #         elif stratification_type == "global":
    #             sampling_function = stratified_sampling

    #         subset_data = sampling_function(
    #             data, 
    #             metric_values,
    #             strat_metric, 
    #             num_bins=num_bins, 
    #             subset_size=subset_size
    #         )

    #         subset_filepath = f"{output_dir}/dataset.jsonl"
    #         save_jsonl(subset_data, subset_filepath)

if __name__ == "__main__":
    main()
