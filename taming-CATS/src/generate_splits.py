import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


COLOR_MAP_FILE = "./../data/colormap/color_map.json"
with open(COLOR_MAP_FILE, "r") as f:
    COLOR_MAP = json.load(f)

def load_jsonl(filepath):
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    
def log_stats(logfile_path, message):
    with open(logfile_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    print(message)

def extract_metrics(data, metrics):
    """Extract specified metric values from dataset."""
    return {metric: [line["source_metrics"].get(metric, 0) for line in data] for metric in metrics}

def remove_outliers_by_char_length(data: list, lower_percentile=3, upper_percentile=97) -> list:
    char_lengths = [len(line["source_text"]) for line in data]
    lower_threshold = np.percentile(char_lengths, lower_percentile)
    upper_threshold = np.percentile(char_lengths, upper_percentile)
    filtered_data = [
        line for line in data
        if lower_threshold <= len(line["source_text"]) <= upper_threshold
    ]
    return filtered_data

def remove_outliers_by_fkgl(data: list, lower_percentile=3, upper_percentile=97) -> list:
    fkgl_values = [line["source_metrics"].get("FKGL", 0) for line in data]
    lower_threshold = np.percentile(fkgl_values, lower_percentile)
    upper_threshold = np.percentile(fkgl_values, upper_percentile)
    filtered_data = [
        line for line in data
        if lower_threshold <= line["source_metrics"].get("FKGL", 0) <= upper_threshold
    ]
    return filtered_data

def remove_outliers_by_ari(data: list, lower_percentile=3, upper_percentile=97) -> list:
    ari_values = [line["source_metrics"].get("ARI", 0) for line in data]
    lower_threshold = np.percentile(ari_values, lower_percentile)
    upper_threshold = np.percentile(ari_values, upper_percentile)
    filtered_data = [
        line for line in data
        if lower_threshold <= line["source_metrics"].get("ARI", 0) <= upper_threshold
    ]
    return filtered_data

def remove_outliers_by_dale_chall(data: list, lower_percentile=3, upper_percentile=97) -> list:
    dale_chall_values = [line["source_metrics"].get("Dale-Chall", 0) for line in data]
    lower_threshold = np.percentile(dale_chall_values, lower_percentile)
    upper_threshold = np.percentile(dale_chall_values, upper_percentile)
    filtered_data = [
        line for line in data
        if lower_threshold <= line["source_metrics"].get("Dale-Chall", 0) <= upper_threshold
    ]
    return filtered_data

def save_jsonl(data, filepath):
    """Save dataset to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in data:
            # TODO remove "original_split" from metadata on the dataset preprocessing level
            # Remove original_split from metadata
            if "metadata" in line and "original_split" in line["metadata"]:
                del line["metadata"]["original_split"]

            f.write(json.dumps(line) + "\n")

def generate_splits(data, metric_values, strat_metric, num_bins=25, seed=None):
    """Generate stratified train, validation, and test splits."""
    values = np.array(metric_values[strat_metric])
    bins = np.histogram_bin_edges(values, bins=num_bins)
    bin_indices = np.digitize(values, bins)
    
    df = pd.DataFrame({"data": data, "bin": bin_indices})
    
    train_data, val_data, test_data = [], [], []

    np.random.seed(seed)

    for bin_id in np.unique(bin_indices):
        bin_data = df[df["bin"] == bin_id]["data"].tolist()
        np.random.shuffle(bin_data)
        train_data.extend(bin_data[:int(len(bin_data) * 0.8)])  # 80% train
        val_data.extend(bin_data[int(len(bin_data) * 0.8):int(len(bin_data) * 0.9)])  # 10% validation
        test_data.extend(bin_data[int(len(bin_data) * 0.9):])  # 10% test

    return train_data, val_data, test_data

def main():
    DATASETS = [
        # "medeasi",
        "newsela",
        # "simpa",
        # "wikilarge_ori_splitwise",
        # "wikilarge_ori_global"
    ]
    DATA_DIR = "./../data"

    # parameters selected through experimentation with distributions (hyperparameter tuning)
    STRAT_METRIC = "FKGL"
    NUM_BINS = 25
    SEED = 42
    MAX_SAMPLES = 3000
    
    for dataset_name in DATASETS:
        print(f"Processing {dataset_name}...")

        SPLIT_SAVE_DIR = f"./../data/splits_new/{dataset_name}_{MAX_SAMPLES}"
        os.makedirs(SPLIT_SAVE_DIR, exist_ok=True)

        LOG_FILE_PATH = os.path.join(SPLIT_SAVE_DIR, "log.txt")
        open(LOG_FILE_PATH, "w").close()  # clear old log
        
        dataset_path = f"{DATA_DIR}/datasets/{dataset_name}/dataset.jsonl"
        data = load_jsonl(dataset_path)

        original_len = len(data)
        log_stats(LOG_FILE_PATH, f"Original data size: {original_len}")

        # remove outliers by key metrics
        for fn, name in [
            (remove_outliers_by_fkgl, "FKGL"),
            (remove_outliers_by_ari, "ARI"),
            (remove_outliers_by_dale_chall, "Dale-Chall"),
            (remove_outliers_by_char_length, "char_length")
        ]:
            before = len(data)
            data = fn(data, lower_percentile=1, upper_percentile=99)
            after = len(data)
            log_stats(LOG_FILE_PATH, f"Removed {before - after} items based on {name} (new size: {after})")

        if len(data) > MAX_SAMPLES:
            np.random.seed(SEED)
            data = list(np.random.choice(data, size=MAX_SAMPLES, replace=False))
            log_stats(LOG_FILE_PATH, f"Randomly sampled {MAX_SAMPLES} items from filtered data")

        full_metric_values = extract_metrics(data, [
                                                "char_count", 
                                                "word_count", 
                                                "sentence_count", 
                                                "FKGL", 
                                                "ARI", 
                                                # "FRE", 
                                                "Dale-Chall"
                                                ]
                                                )

        train_data, val_data, test_data = generate_splits(data, full_metric_values, STRAT_METRIC, num_bins=NUM_BINS, seed=SEED)
        save_jsonl(train_data, f"{SPLIT_SAVE_DIR}/train.jsonl")
        save_jsonl(val_data, f"{SPLIT_SAVE_DIR}/val.jsonl")
        save_jsonl(test_data, f"{SPLIT_SAVE_DIR}/test.jsonl")

        log_stats(LOG_FILE_PATH, f"\nTrain size: {len(train_data)}")
        log_stats(LOG_FILE_PATH, f"Validation size: {len(val_data)}")
        log_stats(LOG_FILE_PATH, f"Test size: {len(test_data)}")
        log_stats(LOG_FILE_PATH, f"Total after split: {len(train_data) + len(val_data) + len(test_data)}")

        print(f"\nSplits for {dataset_name} generated and saved.")

if __name__ == "__main__":
    main()
