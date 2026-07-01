import json
import os
import re
import numpy as np
import argparse

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def log_stats(logfile_path, message):
    with open(logfile_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")
    print(message)

def flatten_jsonl(input_file, output_file, keep_fraction=1.0, seed=42, log_file_path=None):
    flattened_data = []
    total_examples = 0
    total_flattened = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_examples += 1
            example = json.loads(line.strip())
            if isinstance(example.get("simplifications", None), list):
                for simplification in example["simplifications"]:
                    total_flattened += 1
                    flattened_data.append({
                        "global_id": example["global_id"],
                        "source_text": clean_text(example.get("source_text", "")),
                        "source_metrics": example.get("source_metrics", {}),
                        "metadata": example["metadata"],
                        "simplification_text": clean_text(simplification.get("simplification_text", "")),
                        "target_metrics": simplification.get("target_metrics", {}),
                        "simplification_version": simplification.get("simplification_version", ""),
                        "grade_level": simplification.get("grade_level", ""),
                        "simplification_dimensions": simplification.get("simplification_dimensions", {})
                    })

            else:
                # If 'simplifications' is missing or not a list, handle it here
                print(f"Warning: 'simplifications' missing or not a list in example: {example}")
    

    sample_size = max(1, int(keep_fraction * len(flattened_data)))
    np.random.seed(seed)
    flattened_data = list(np.random.choice(flattened_data, size=sample_size, replace=False))

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in flattened_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log_stats(log_file_path, f"Input file: {input_file}")
    log_stats(log_file_path, f"Total examples read: {total_examples}")
    log_stats(log_file_path, f"Total simplifications flattened: {total_flattened}")
    log_stats(log_file_path, "-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Flatten JSONL files and sample a fraction of the data.")
    parser.add_argument("--keep_fraction", type=float, default=1.0, help="Fraction of data to keep after flattening.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to process.")
    args = parser.parse_args()


    print(f"{5*"*"} Processing {args.dataset}... {5*"*"}")
    input_dir = f"./data/splits_new/{args.dataset}"
    output_dir = f"./data/splits_flattened/{args.dataset}_{args.keep_fraction}"
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, "log.txt")
    open(log_file_path, "w").close()

    for split in ["train", "val", "test"]:
        input_jsonl = f'{input_dir}/{split}.jsonl'
        output_jsonl = f'{output_dir}/{split}.jsonl'

        flatten_jsonl(input_jsonl, output_jsonl, args.keep_fraction, seed=args.seed, log_file_path=log_file_path)

    log_stats(log_file_path, f"Flattening completed for dataset: {args.dataset}")

if __name__=="__main__":
    main()
