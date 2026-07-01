import json
import os
import numpy as np
import pandas as pd
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
            # Compute missing metrics for source text
            for simplification in line["simplifications"]:
                bleu = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bleu()
                bertscore = Metrics(
                    input_text=simplification["simplification_text"], 
                    source_text=source_text
                    ).compute_bertscore()
                
                simplification['target_metrics']['BLEU'] = bleu
                simplification['target_metrics']['BERTScore'] = bertscore

            f.write(json.dumps(line) + "\n")

def extract_metrics(data: list, metrics: list) -> dict:
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def stratified_sampling(data: list, metric_values: dict, metric: str, num_bins=20, subset_size=2000) -> list:
    """Perform stratified sampling based on a single metric."""
    values = np.array(metric_values[metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=subset_size / len(df), random_state=42))
    return subset["data"].tolist()

def stratified_sampling_from_split(data: list, metric_values: dict, metric: str, num_bins=20, subset_size=2000):
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

    def sample_split(split_data: list, split_metrics: dict, split_size: int, drop_outliers=True) -> list:
        """Helper function to perform stratified sampling for each split."""
        if len(split_data) == 0:
            return []  # In case a split has no data (edge case)
        values = np.array(split_metrics[metric])

        # Remove outliers
        if drop_outliers:
            lower, upper = np.percentile(values, [5, 95])
            mask = (values >= lower) & (values <= upper)
            values = values[mask]
            split_data = [d for i, d in enumerate(split_data) if mask[i]]

        bins = np.histogram_bin_edges(values, bins=num_bins)
        bin_indices = np.digitize(values, bins)
        df = pd.DataFrame({"data": split_data, "bin": bin_indices})
        subset = df.groupby("bin", group_keys=False).apply(lambda x: x.sample(frac=split_size / len(df), random_state=42))
        return subset["data"].tolist()

    sampled_train = sample_split(train_data, train_metrics, train_size)
    sampled_valid = sample_split(valid_data, valid_metrics, valid_size)
    sampled_test = sample_split(test_data, test_metrics, test_size)

    return sampled_train + sampled_valid + sampled_test

def main():
    # Parameters

    num_bins = 25
    stratification_metric = "FKGL"
    subset_size = 2000
    base_dir = "./../data/datasets"
    experiment_name = f"wikilarge_ori_global_{subset_size}"
    full_dataset_path = f"{base_dir}/wikilarge_ori/dataset.jsonl"
    
    # Load dataset
    data = load_jsonl(full_dataset_path)
    metrics = ["char_count", "word_count", "FKGL", "ARI", "FRE", "Dale-Chall"]
    metric_values = extract_metrics(data, metrics)

    # splitwise_output_dir = f"{base_dir}/{experiment_name}"
    # os.makedirs(splitwise_output_dir, exist_ok=True)
    # splitwise_output_path = f"{splitwise_output_dir}/dataset.jsonl"

    # sampled_splitwise = stratified_sampling_from_split(
    #     data, 
    #     metric_values, 
    #     stratification_metric, 
    #     num_bins=num_bins, 
    #     subset_size=subset_size
    # )
    # save_jsonl(sampled_splitwise, splitwise_output_path)
    # print(f"Saved splitwise stratified dataset to {splitwise_output_path}.")

    global_output_dir = f"{base_dir}/wikilarge_ori_global_{subset_size}"
    os.makedirs(global_output_dir, exist_ok=True)
    global_output_path = f"{global_output_dir}/dataset.jsonl"

    # Stratified Sampling - Global
    sampled_global = stratified_sampling(
        data, 
        metric_values, 
        stratification_metric, 
        num_bins=num_bins, 
        subset_size=subset_size
    )
    save_jsonl(sampled_global, global_output_path)
    print(f"Saved global stratified dataset to {global_output_dir}.")

if __name__ == "__main__":
    main()
