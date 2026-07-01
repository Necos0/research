import json
import pandas as pd
from classes.Dataset import Newsela, MedEASi, WikiLarge, SimPALex, SimPASyn
from classes.Metrics import Metrics
import os

DATA_DIR = "./../data"

def load_dataset(dataset_class):
    dataset = dataset_class(limit=None)
    # dataset = dataset_class(limit=10)
    dataset.load_data()

    if dataset.data_df.empty:
        raise ValueError(f"No data loaded for dataset: {dataset.dataset_name}")

     # Filter out non-English instances for Newsela
    if dataset.dataset_name == "newsela":
        dataset.data_df = dataset.data_df[dataset.data_df[dataset.language] == "en"]
    return dataset, dataset.data_df.groupby(dataset.grouping_tag)

def find_source_text(dataset, group):
    if dataset.dataset_name in ["newsela", "simpa_lexical", "simpa_syntactic"]:
        source_row = group[group[dataset.simplification_version] == 0]
        source_text = source_row.iloc[0][dataset.text] if not source_row.empty else ""
    elif dataset.dataset_name in ["medeasi", "wikilarge", "wikilarge_ori"]:
        source_row = group.iloc[[0]]
        source_text = source_row.iloc[0][dataset.source_text]
    else:
        raise ValueError(f"Unsupported dataset: {dataset.dataset_name}")
    
    source_metrics = Metrics(
                            input_text=source_text
                            ).compute_metrics()
    return source_text, source_metrics

def find_language(dataset, group):
    return group.iloc[0][dataset.language] if dataset.dataset_name == "newsela" else dataset.language

def find_split(dataset, group):
    return "" if dataset.dataset_name in ["newsela", "simpa_lexical", "simpa_syntactic"] else group.iloc[0][dataset.split]

def convert_to_jsonl(dataset, grouped_df):

    jsonl_data = []
    counter = 1

    for _, group in grouped_df:
        source_text, source_metrics = find_source_text(dataset, group)
        language = find_language(dataset, group)
        split = find_split(dataset, group)

        ######## source ########
        json_entry = {
            "global_id": f"{dataset.dataset_name}_{counter:06d}",
            "source_text": source_text,
            "metadata": {
                "dataset": dataset.dataset_name,
                "original_split": split,
                "experiment_split": "", # TODO
                "domain": dataset.domain,
                "language": language, 
                "annotation": dataset.annotation,
                "level": dataset.alignment_level
            },
            "source_metrics": {
                "FRE": source_metrics["FRE"],
                "ARI": source_metrics["ARI"],
                "FKGL": source_metrics["FKGL"],
                "Dale-Chall": source_metrics["Dale-Chall"],
                "char_count": source_metrics["char_count"],
                "alphanum_count": source_metrics["alphanum_count"],
                "word_count": source_metrics["word_count"],
                "sentence_count": source_metrics["sent_count"]
            },
            "simplifications": []
        }

        ######## target ########
        for _, row in group.iterrows():
            if dataset.dataset_name in ["newsela", "simpa_lexical", "simpa_syntactic"] and row[dataset.simplification_version] == 0:
                continue
            if dataset.dataset_name in ["newsela", "simpa_lexical", "simpa_syntactic"]:
                simplification_text = row[dataset.text]
            else:
                simplification_text = row[dataset.target_text]
            simplification_metrics = Metrics(
                                            input_text=simplification_text, 
                                            source_text=source_text
                                            ).compute_metrics()
            # control_tokens_access = get_control_tokens_access()
            # control_tokens_muss = get_control_tokens_muss()

            simplification_entry = {
                "simplification_text": simplification_text,
                "simplification_version": 1 if dataset.dataset_name in ["medeasi", "wikilarge", "wikilarge_ori", "simpa_syntactic"] else int(row[dataset.simplification_version]),
                "grade_level": int(row[dataset.grade_level]) if dataset.dataset_name == "newsela" else -1,
                # "control_attributes_access": control_tokens_access,
                # "control_attributes_muss": control_tokens_muss,
                "simplification_dimensions": {
                    "linguistic": {
                        "lexical": dataset.dataset_name == "simpa_lexical",
                        "syntactic": dataset.dataset_name == "simpa_syntactic"
                    }
                },
                "target_metrics": {
                    "FRE": simplification_metrics["FRE"],
                    "ARI": simplification_metrics["ARI"],
                    "FKGL": simplification_metrics["FKGL"],
                    "Dale-Chall": simplification_metrics["Dale-Chall"],
                    "char_count": simplification_metrics["char_count"],
                    "alphanum_count": simplification_metrics["alphanum_count"],
                    "word_count": simplification_metrics["word_count"],
                    "sentence_count": simplification_metrics["sent_count"],
                    # avoid division by zero
                    "char_compression_rate": round(simplification_metrics["char_count"]/source_metrics["char_count"], 2) if source_metrics["char_count"] > 0 else 0.0,
                    "word_compression_rate": round(simplification_metrics["word_count"]/source_metrics["word_count"], 2) if source_metrics["word_count"] > 0 else 0.0,
                    "sentence_compression_rate": round(simplification_metrics["sent_count"]/source_metrics["sent_count"], 2)if source_metrics["sent_count"] > 0 else 0.0,
                    # compute BLUE and BERTScore later for wikilarge, once working with a subset
                    "BLEU": simplification_metrics["BLEU"],
                    # "BLEU": 0.0,
                    "BERTScore": simplification_metrics["BERTScore"], 
                    # "BERTScore": 0.0 
                }
            }

            json_entry["simplifications"].append(simplification_entry)
            print(f"Processed {dataset.dataset_name} - {counter:06d}")

        jsonl_data.append(json_entry)
        counter += 1

    return jsonl_data

def save_jsonl(dataset, jsonl_data):

    output_dir = f"{DATA_DIR}/datasets/{dataset.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/dataset.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")

    print(f"JSONL file saved: {output_file}")
    print(f"Total entries: {len(jsonl_data)}")

def main():

    # print("Processing SimPA (syntactic simplifications)...")
    # dataset, dataset_grouped = load_dataset(SimPASyn)
    # jsonl_data = convert_to_jsonl(dataset, dataset_grouped)
    # save_jsonl(dataset, jsonl_data)
    # print("SimPA (syntactic simplifications) successfuly saved to jsonl.\n***\n")

    # print("Processing SimPA (lexical simplifications)...")
    # dataset, dataset_grouped = load_dataset(SimPALex)
    # jsonl_data = convert_to_jsonl(dataset, dataset_grouped)
    # save_jsonl(dataset, jsonl_data)
    # print("SimPA (lexical simplifications) successfuly saved to jsonl.\n***\n")

    print("Processing Newsela...")
    dataset, dataset_grouped = load_dataset(Newsela)
    jsonl_data = convert_to_jsonl(dataset, dataset_grouped)
    save_jsonl(dataset, jsonl_data)
    print("Newsela successfuly saved to jsonl.\n***\n")

    # print("Processing MedEASi...")
    # dataset, dataset_grouped = load_dataset(MedEASi)
    # jsonl_data = convert_to_jsonl(dataset, dataset_grouped)
    # save_jsonl(dataset, jsonl_data)
    # print("MedEASi successfully saved to jsonl.\n***\n")

    # print("Processing WikiLarge...")
    # dataset, dataset_grouped = load_dataset(WikiLarge)
    # jsonl_data = convert_to_jsonl(dataset, dataset_grouped)
    # save_jsonl(dataset, jsonl_data)
    # print("WikiLarge successfully saved to jsonl.\n---\n")

    print("\nAll datasets have been successfully processed.")



if __name__=="__main__":
    main()