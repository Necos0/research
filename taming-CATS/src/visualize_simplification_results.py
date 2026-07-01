import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from classes.Metrics import Metrics


def collect_scores(input_file, metric):
    scores = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)

            # Extract metric values
            source_val = entry["source_metrics"][metric]
            reference_val = entry["reference_metrics"][metric]
            target_val = entry["target_metrics"][metric]

            # Store values in lists
            scores["Source"].append(source_val)
            scores["Reference"].append(reference_val)
            scores["Target"].append(target_val)

    # Convert to a structured format (for boxplot + trajectory visualization)
    data = []
    trajectories = []

    for idx, (source, reference, target) in enumerate(zip(scores["Source"], scores["Reference"], scores["Target"])):
        # Order for plotting: Source, Reference, Target
        data.extend([("Source", source), ("Reference", reference), ("Target", target)])
        trajectories.append((idx, [source, reference, target]))  # Keep order for trajectory lines

    return data, trajectories

def plot_scores_violin(data, metric, save_dir, color_map):
    colors = {
        "Source": color_map["text_type"]["source"],
        "Reference": color_map["text_type"]["reference"],
        "Target": color_map["text_type"]["target"],
    }

    plt.figure(figsize=(5, 5))

    # Create a violin plot
    ax = sns.violinplot(
        x=[x[0] for x in data], 
        y=[x[1] for x in data], 
        palette=colors,
        inner="quart",  # Show quartiles and individual data points
        linewidth=1.25,
    )

    plt.ylabel(f"{metric}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{metric}_violin.png", dpi=400)
    # plt.show()


def plot_ref_vs_target_similarity(input_file, save_dir, color_map):
    
    # Initialize lists to store BLEU and BERTScore values
    bleu_target = []
    bleu_reference = []
    bertscore_target = []
    bertscore_reference = []

    # Read the data from the file
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)

            # Extract BLEU and BERTScore values for target and reference
            bleu_target_val = entry["target_metrics"].get("BLEU", None)
            bleu_reference_val = entry["reference_metrics"].get("BLEU", None)
            bertscore_target_val = entry["target_metrics"].get("BERTScore", None)
            bertscore_reference_val = entry["reference_metrics"].get("BERTScore", None)

            # Append values if they exist
            if bleu_target_val is not None and bleu_reference_val is not None:
                bleu_target.append(bleu_target_val)
                bleu_reference.append(bleu_reference_val)

            if bertscore_target_val is not None and bertscore_reference_val is not None:
                bertscore_target.append(bertscore_target_val)
                bertscore_reference.append(bertscore_reference_val)

    # Create a figure with two subplots (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Get the min and max values for scaling
    min_bleu = min(min(bleu_target), min(bleu_reference))
    max_bleu = max(max(bleu_target), max(bleu_reference))
    
    min_bertscore = min(min(bertscore_target), min(bertscore_reference))
    max_bertscore = max(max(bertscore_target), max(bertscore_reference))

    # Plot 1: Reference BLEU vs Target BLEU
    sns.scatterplot(x=bleu_reference, y=bleu_target, color=color_map["metrics"]["BLEU"], alpha=0.7, s=100, ax=axes[0])
    sns.regplot(x=bleu_reference, y=bleu_target, scatter=False, ax=axes[0], color=color_map["metrics"]["BLEU"], line_kws={"color": "black", "lw": 2, "ls": "--"})
    axes[0].set_xlabel("Reference BLEU Score")
    axes[0].set_ylabel("Target BLEU Score")
    axes[0].set_xlim(min_bleu, max_bleu)  # Set the same scale for both axes
    axes[0].set_ylim(min_bleu, max_bleu)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Plot 2: Reference BERTScore vs Target BERTScore
    sns.scatterplot(x=bertscore_reference, y=bertscore_target, color=color_map["metrics"]["BERTScore"], alpha=0.7, s=100, ax=axes[1])
    sns.regplot(x=bertscore_reference, y=bertscore_target, scatter=False, ax=axes[1], color=color_map["metrics"]["BERTScore"], line_kws={"color": "black", "lw": 2, "ls": "--"})
    axes[1].set_xlabel("Reference BERTScore")
    axes[1].set_ylabel("Target BERTScore")
    axes[1].set_xlim(min_bertscore, max_bertscore)  # Set the same scale for both axes
    axes[1].set_ylim(min_bertscore, max_bertscore)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ref_vs_target_similarity.png", dpi=400)
    # plt.show()


def main():

    with open("./../data/colormap/color_map.json", "r", encoding="utf-8") as file:
        color_map = json.load(file)

    control_token_metric = "ARI" # TODO
    user_prompt_id = "token_explanation_examples" # TODO select "token_explanation_examples", "token_explanation", "token", "no_token"

    experiment_dir = f"./../experiments/prompting/dynamic_prompting_{user_prompt_id}_{control_token_metric}/vanilla/gpt-4o-mini/medeasi/"
    # find latest run in the directory
    runs = os.listdir(experiment_dir)
    runs.sort()
    run = runs[-1]
    print(f"Selecting latest run: {run}")

    

    input_file = os.path.join(experiment_dir, run, "results.jsonl")
    output_dir = os.path.join(experiment_dir, run, "plots/")
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = ["ARI", "FKGL", "Dale-Chall", "char_count", "word_count"]

    for metric in metrics_to_plot:
        data, _ = collect_scores(input_file=input_file, metric=metric)  # Collect fresh data for each metric
        plot_scores_violin(data=data, metric=metric, save_dir=output_dir, color_map=color_map)

    plot_ref_vs_target_similarity(input_file=input_file, save_dir=output_dir, color_map=color_map)
    

if __name__ == "__main__":
    main()