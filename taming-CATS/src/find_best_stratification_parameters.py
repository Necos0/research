import json
import pandas as pd

# Store the results across all datasets
results = {}

# Path to your experiment results file
EXPERIMENT_NAME = "splits_sampling"
RESULTS_FILE = f"./../experiments/{EXPERIMENT_NAME}/all_results.json"

with open(RESULTS_FILE, "r") as f:
    experiment_results = json.load(f)

# Convert results to DataFrame
df = pd.DataFrame(experiment_results)

# Log files where we will save the best results
LOG_FILE_PER_DATASET = f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_per_dataset.txt"
LOG_FILE_ACROSS_DATASETS = f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_across_datasets.txt"

# Results per dataset
with open(LOG_FILE_PER_DATASET, "w") as log_file:
    log_file.write("Best Stratification Results (Per Dataset)\n")
    log_file.write("="*50 + "\n\n")
    log_file.write("Note: For each dataset, the best stratification metric and number of bins are "
                   "determined based on the lowest average KS divergence. All seed runs (10 seeds)"
                   "and averaged the KS divergence values. The metric with the lowest average KS divergence "
                   "across all combinations of stratification metric and number of bins is chosen.\n")
    log_file.write("="*50 + "\n\n")
    
    # Per dataset: Find the best result for each stratification metric and number of bins, for each dataset
    for dataset in df["dataset"].unique():
        log_file.write(f"\nProcessing dataset: {dataset}\n")
        log_file.write("-"*50 + "\n")
        
        dataset_df = df[df["dataset"] == dataset]
        
        # Find the best result for each stratification metric and number of bins
        best_results_per_metric = dataset_df.groupby(["strat_metric", "num_bins"]).agg(
            mean_KS=("average_KS", "mean"),
            std_KS=("average_KS", "std")
        ).reset_index()
        
        # Find the best stratification metric (lowest mean KS) and best num_bins for that metric
        best_result = best_results_per_metric.loc[best_results_per_metric["mean_KS"].idxmin()]
        
        best_strat_metric = best_result["strat_metric"]
        best_num_bins = int(best_result["num_bins"])
        best_avg_ks = best_result["mean_KS"]
        
        # Write best results for this dataset to log file
        log_file.write(f"Best Stratification Metric: {best_strat_metric}\n")
        log_file.write(f"Best Number of Bins: {best_num_bins}\n")
        log_file.write(f"Lowest Average KS Divergence: {best_avg_ks:.6f}\n")
        log_file.write("-"*50 + "\n")
        
        # Save the best stratification results for this dataset
        results[dataset] = {
            "best_strat_metric": best_strat_metric,
            "best_num_bins": best_num_bins,
            "best_avg_ks": best_avg_ks
        }

# Save per dataset results in JSON
with open(f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_per_dataset.json", "w") as f:
    json.dump(results, f, indent=4)

# Results across all datasets
with open(LOG_FILE_ACROSS_DATASETS, "w") as log_file:
    log_file.write("Best Stratification Results (Across All Datasets)\n")
    log_file.write("="*50 + "\n\n")
    log_file.write("Note: The best stratification metric and number of bins across all datasets were "
                   "determined by aggregating the average KS divergence values across all datasets. "
                   "The combination with the lowest average KS divergence across all datasets is considered the best.\n")
    log_file.write("="*50 + "\n\n")
    
    # Aggregate the data from all datasets and find the best stratification metric and bins across all datasets
    all_results = df.groupby(["strat_metric", "num_bins"]).agg(
        mean_KS=("average_KS", "mean"),
        std_KS=("average_KS", "std")
    ).reset_index()

    # Find the best stratification metric (lowest mean KS) across all datasets
    best_result = all_results.loc[all_results["mean_KS"].idxmin()]
    
    best_strat_metric = best_result["strat_metric"]
    best_num_bins = int(best_result["num_bins"])
    best_avg_ks = best_result["mean_KS"]
    
    # Write best results across all datasets to log file
    log_file.write(f"Best Stratification Metric (Across All Datasets): {best_strat_metric}\n")
    log_file.write(f"Best Number of Bins: {best_num_bins}\n")
    log_file.write(f"Lowest Average KS Divergence: {best_avg_ks:.6f}\n")
    log_file.write("-"*50 + "\n")
    
    # Save the best stratification results across all datasets in the results dictionary
    results["best_strat_metric_across_datasets"] = best_strat_metric
    results["best_num_bins_across_datasets"] = best_num_bins
    results["best_avg_ks_across_datasets"] = best_avg_ks

# Save the results across all datasets in JSON
with open(f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results_across_datasets.json", "w") as f:
    json.dump(results, f, indent=4)

# Output message
print(f"Best stratification results per dataset saved to {LOG_FILE_PER_DATASET}")
print(f"Best stratification results across all datasets saved to {LOG_FILE_ACROSS_DATASETS}")
