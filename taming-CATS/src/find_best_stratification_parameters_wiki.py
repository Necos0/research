import json
import pandas as pd

results = {}

EXPERIMENT_NAME = "splits_no_dev_no_wikilarge"
RESULTS_FILE = f"./../experiments/{EXPERIMENT_NAME}/all_results.json"

with open(RESULTS_FILE, "r") as f:
    experiment_results = json.load(f)

df = pd.DataFrame(experiment_results)

for metric in df["strat_metric"].unique():
    # Filter results for the current metric
    metric_df = df[df["strat_metric"] == metric]

    # Find the best result for each num_bins value based on the lowest average_KS
    best_per_bin = metric_df.loc[metric_df.groupby("num_bins")["average_KS"].idxmin()]

    # Find the best bin with the lowest average_KS value
    best_bin = best_per_bin.loc[best_per_bin["average_KS"].idxmin()]

    # Store the best results for the metric
    results[metric] = {
        "best_num_bins": int(best_bin["num_bins"]),
        "average_KS": float(best_bin["average_KS"]),
    }

# Save the results to a file
with open(f"./../experiments/{EXPERIMENT_NAME}/best_stratification_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Determine the best overall stratification metric based on the lowest average_KS
best_metric = None
best_num_bins = None
best_ks = float("inf")

# Iterate over each stratification metric to find the best one
for strat_metric, values in results.items():
    avg_ks = values["average_KS"]  # Extract KS divergence
    num_bins = values["best_num_bins"]  # Extract best num_bins

    if avg_ks < best_ks:  # Find the minimum KS divergence
        best_ks = avg_ks
        best_metric = strat_metric
        best_num_bins = num_bins  # Assign correct num_bins value

# Output the best stratification results
print(f"Best stratification metric: {best_metric}")
print(f"Best number of bins: {best_num_bins}")
print(f"Lowest average KS divergence: {best_ks:.6f}")