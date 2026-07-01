import json
import os
import random
random.seed(42)

def load_jsonl(filepath: str) -> list:
    """Load dataset from a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
def save_jsonl(filepath: str, data: list) -> None:
    """Save dataset as a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def combine_datasets(dataset_paths: list, output_path: str) -> None:
    # combine train sets
    train_data = []
    for dataset_path in dataset_paths:
        train_data += load_jsonl(os.path.join(dataset_path, "train.jsonl"))
    # combine test sets
    test_data = []
    for dataset_path in dataset_paths:
        test_data += load_jsonl(os.path.join(dataset_path, "test.jsonl"))
    # combine val sets
    val_data = []
    for dataset_path in dataset_paths:
        val_data += load_jsonl(os.path.join(dataset_path, "val.jsonl"))
    return train_data, test_data, val_data

def main():

    tasks = ["flattened", "flattened_hq"]
    flattened_dir = "splits_flattened"
    flattened_hq_dir = "splits_flattened_filtered"
    
    dataset_path = None

    for task in tasks:

        if task == "flattened":
            dataset_path = flattened_dir
        elif task == "flattened_hq":
            dataset_path = flattened_hq_dir

        medeasi_dir_path = f"./../data/{dataset_path}/medeasi"
        simpa_dir_path = f"./../data/{dataset_path}/simpa"
        wikilarge_splitwise_dir_path = f"./../data/{dataset_path}/wikilarge_ori_splitwise"
        dataset_paths = [medeasi_dir_path, simpa_dir_path, wikilarge_splitwise_dir_path]

        output_dir = f"./../data/{dataset_path}/combined"
        os.makedirs(output_dir, exist_ok=True)

        train, test, val = combine_datasets(dataset_paths, output_dir)
        random.shuffle(train)
        random.shuffle(test)
        random.shuffle(val)

        save_jsonl(os.path.join(output_dir, "train.jsonl"), train)
        save_jsonl(os.path.join(output_dir, "test.jsonl"), test)
        save_jsonl(os.path.join(output_dir, "val.jsonl"), val)
        print(f"Combined dataset saved to {output_dir} with {len(train)} train, {len(test)} test, and {len(val)} val entries.")


if __name__ == "__main__":
    main()