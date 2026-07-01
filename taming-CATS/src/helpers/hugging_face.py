from datasets import load_dataset
import os

def load_dataset_from_hf(DATASET, split="validation", slice=-1) -> list:

    # ローカルに整形済みデータがあればそれを優先（研究室サーバーでのオフライン再現用）。
    # 無ければ従来どおり著者の HF Hub (shtosti/<DATASET>) から取得する。
    local_dir = os.path.join("data/splits_flattened_filtered", DATASET)
    if os.path.isdir(local_dir):
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(local_dir, "train.jsonl"),
                "test": os.path.join(local_dir, "test.jsonl"),
                "validation": os.path.join(local_dir, "val.jsonl"),
            },
        )
    else:
        hf_token = os.getenv("HF_TOKEN")
        os.system(f"hf auth login --token {hf_token}")

        dataset = load_dataset(
            f"shtosti/{DATASET}",
            data_files={"train": "train.jsonl", "test": "test.jsonl", "validation": "val.jsonl"}
        )
    if split not in dataset:
        raise ValueError(f"Split {split} not found in dataset")
    
    split_dataset = dataset[split]
    if slice > 0:
        split_dataset = split_dataset.select(range(slice))

    print(f"{split}:\t Inspect the dataset format:\n", split_dataset[:1])
    
    return split_dataset

def get_model_short_name(model_name):
    return model_name.split("/")[-1]