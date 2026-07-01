# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
# import sys

# # TODO select the correct dataset name 
# # DATASET_NAME = "newsela"
# # DATASET_NAME = "medeasi"
# DATASET_NAME = "wikilarge"

# DATA_DIR = "./../data" 
# DATASET_DIR = f"{DATA_DIR}/datasets/{DATASET_NAME}"
# DATASET_PATH = f"{DATASET_DIR}/dataset.jsonl"

# def load_jsonl(filepath):
#     data = []
#     with open(filepath, "r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def keep_eng(data):
#     return [entry for entry in data if entry["metadata"]["language"] == "en"]

# def compute_statistics(data):
#     num_sources = len(data)
#     num_simplifications = sum(len(entry["simplifications"]) for entry in data)
    
#     # Grade level distribution
#     grade_levels = [simp["grade_level"] for entry in data for simp in entry["simplifications"]]
#     grade_level_counts = Counter(grade_levels)
    
#     # Language distribution
#     languages = [entry["metadata"]["language"] for entry in data]
#     language_counts = Counter(languages)
    
#     # Text length stats
#     source_char_counts = [entry["source_metrics"]["char_count"] for entry in data]
#     simplification_char_counts = [simp["target_metrics"]["char_count"] for entry in data for simp in entry["simplifications"]]
#     source_word_counts = [entry["source_metrics"]["word_count"] for entry in data]
#     simplification_word_counts = [simp["target_metrics"]["word_count"] for entry in data for simp in entry["simplifications"]]
#     source_sent_counts = [entry["source_metrics"]["sentence_count"] for entry in data]
#     simplification_sent_counts = [simp["target_metrics"]["sentence_count"] for entry in data for simp in entry["simplifications"]]

#     # compression rates
#     char_compression_rate = [simp["target_metrics"]["char_compression_rate"] for entry in data for simp in entry["simplifications"]]
#     word_compression_rate = [simp["target_metrics"]["word_compression_rate"] for entry in data for simp in entry["simplifications"]]
#     sent_compression_rate = [simp["target_metrics"]["sentence_compression_rate"] for entry in data for simp in entry["simplifications"]]

#     # Readability scores
#     # source
#     source_fkgl_scores = [entry["source_metrics"]["FKGL"] for entry in data]
#     source_fre_scores = [entry["source_metrics"]["FRE"] for entry in data]
#     source_ari_scores = [entry["source_metrics"]["ARI"] for entry in data]
#     source_dale_chall_scores = [entry["source_metrics"]["Dale-Chall"] for entry in data]
#     # target
#     target_fkgl_scores = [simp["target_metrics"]["FKGL"] for entry in data for simp in entry["simplifications"]]
#     target_fre_scores = [simp["target_metrics"]["FRE"] for entry in data for simp in entry["simplifications"]]
#     target_ari_scores = [simp["target_metrics"]["ARI"] for entry in data for simp in entry["simplifications"]]
#     target_dale_chall_scores = [simp["target_metrics"]["Dale-Chall"] for entry in data for simp in entry["simplifications"]]
    
#     stats = {
#         "n source texts": num_sources,
#         "n simplifications (total)": num_simplifications,
#         "mean n simplifications per source text": round(num_simplifications/num_sources, 2),
#         "Grade level distribution": grade_level_counts,
#         "Language distribution": language_counts,
#         # average words
#         "Average words per source text": sum(source_word_counts) / len(source_word_counts),
#         "Average words per simplification": sum(simplification_word_counts) / len(simplification_word_counts),
#         # average sentences
#         "Average sentences per source text": sum(source_sent_counts) / len(source_sent_counts),
#         "Average sentences per simplification": sum(simplification_sent_counts) / len(simplification_sent_counts),
#         # average compression rates
#         "Average char compression rate": sum(char_compression_rate) / len(char_compression_rate),
#         "Average word compression rate": sum(word_compression_rate) / len(word_compression_rate),
#         "Average sentence compression rate": sum(sent_compression_rate) / len(sent_compression_rate),
#         # metrics
#         # FKGL
#         "Source Average FKGL score": sum(source_fkgl_scores) / len(source_fkgl_scores),
#         "Target Average FKGL score": sum(target_fkgl_scores) / len(target_fkgl_scores),
#         # FRE
#         "Source Average FRE score": sum(source_fre_scores) / len(source_fre_scores),
#         "Target Average FRE score": sum(target_fre_scores) / len(target_fre_scores),
#         # ARI
#         "Source Average ARI score": sum(source_ari_scores) / len(source_ari_scores),
#         "Target Average ARI score": sum(target_ari_scores) / len(target_ari_scores),
#         # Dale-Chall
#         "Source Average Dale-Chal score": sum(source_dale_chall_scores) / len(source_dale_chall_scores),
#         "Target Average Dale-Chal score": sum(target_dale_chall_scores) / len(target_dale_chall_scores),
#     }
    
#     return stats

# def compute_target_metrics_by_grade(data):

#     metrics_by_grade = {}

#     for entry in data:
#         for simp in entry["simplifications"]:
#             grade = simp["grade_level"]
#             if grade not in metrics_by_grade:
#                 metrics_by_grade[grade] = {"FKGL": [], "FRE": [], "ARI": [], "Dale-Chall": [], "words": [], "sentences": [], "characters": []}

#             metrics_by_grade[grade]["FKGL"].append(simp["target_metrics"]["FKGL"])
#             metrics_by_grade[grade]["FRE"].append(simp["target_metrics"]["FRE"])
#             metrics_by_grade[grade]["ARI"].append(simp["target_metrics"]["ARI"])
#             metrics_by_grade[grade]["Dale-Chall"].append(simp["target_metrics"]["Dale-Chall"])
#             metrics_by_grade[grade]["words"].append(simp["target_metrics"]["word_count"])
#             metrics_by_grade[grade]["sentences"].append(simp["target_metrics"]["sentence_count"])
#             metrics_by_grade[grade]["characters"].append(simp["target_metrics"]["char_count"])

#     # Compute averages
#     avg_metrics = {
#         grade: {
#             "FKGL": sum(values["FKGL"]) / len(values["FKGL"]),
#             "FRE": sum(values["FRE"]) / len(values["FRE"]),
#             "ARI": sum(values["ARI"]) / len(values["ARI"]),
#             "Dale-Chall": sum(values["Dale-Chall"]) / len(values["Dale-Chall"]),
#             "words": sum(values["words"]) / len(values["words"]),
#             "sentences": sum(values["sentences"]) / len(values["sentences"]),
#             "characters": sum(values["characters"]) / len(values["characters"])
#         }
#         for grade, values in metrics_by_grade.items()
#     }

#     return avg_metrics

# def compute_target_metrics_average(data):
#     target_word_counts = []
#     target_sentence_counts = []
#     target_char_counts = []
#     target_fkgl_scores = []
#     target_fre_scores = []
#     target_ari_scores = []
#     target_dale_chall_scores = []

#     # Iterate through all the data and simplifications
#     for entry in data:
#         for simp in entry["simplifications"]:
#             target_word_counts.append(simp["target_metrics"]["word_count"])
#             target_sentence_counts.append(simp["target_metrics"]["sentence_count"])
#             target_char_counts.append(simp["target_metrics"]["char_count"])
#             target_fkgl_scores.append(simp["target_metrics"]["FKGL"])
#             target_fre_scores.append(simp["target_metrics"]["FRE"])
#             target_ari_scores.append(simp["target_metrics"]["ARI"])
#             target_dale_chall_scores.append(simp["target_metrics"]["Dale-Chall"])

#     # Calculate the averages for each metric
#     avg_target_metrics = {
#         "word_count": sum(target_word_counts) / len(target_word_counts),
#         "sentence_count": sum(target_sentence_counts) / len(target_sentence_counts),
#         "char_count": sum(target_char_counts) / len(target_char_counts),
#         "FKGL": sum(target_fkgl_scores) / len(target_fkgl_scores),
#         "FRE": sum(target_fre_scores) / len(target_fre_scores),
#         "ARI": sum(target_ari_scores) / len(target_ari_scores),
#         "Dale-Chall": sum(target_dale_chall_scores) / len(target_dale_chall_scores),
#     }

#     return avg_target_metrics

# def plot_target_metrics_by_grade(avg_metrics):
#     grades = sorted(avg_metrics.keys())
#     fkgl_scores = [avg_metrics[grade]["FKGL"] for grade in grades]
#     fre_scores = [avg_metrics[grade]["FRE"] for grade in grades]
#     ari_scores = [avg_metrics[grade]["ARI"] for grade in grades]
#     dale_chall_scores = [avg_metrics[grade]["Dale-Chall"] for grade in grades]
#     words = [avg_metrics[grade]["words"] for grade in grades]
#     sentences = [avg_metrics[grade]["sentences"] for grade in grades]
#     characters = [avg_metrics[grade]["characters"] for grade in grades]

#     # ARI, FKGL, FRE, Dale-Chall
#     plt.figure(figsize=(10, 5))
#     plt.plot(grades, fkgl_scores, marker="o", label="FKGL")
#     plt.plot(grades, fre_scores, marker="s", label="FRE")
#     plt.plot(grades, ari_scores, marker="^", label="ARI")
#     plt.plot(grades, dale_chall_scores, marker="*", label="Dale-Chall")
#     plt.xlabel("Grade Level")
#     plt.ylabel("Score")
#     plt.title("Readability by Grade Level")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{DATASET_DIR}/ARI_FKGL_FRE_DC.png", dpi=400, bbox_inches="tight")
#     # plt.show()
#     plt.close()

#     # without FRE (for better scaling)
#     plt.figure(figsize=(10, 5))
#     plt.plot(grades, fkgl_scores, marker="o", label="FKGL")
#     plt.plot(grades, ari_scores, marker="^", label="ARI")
#     plt.plot(grades, dale_chall_scores, marker="*", label="Dale-Chall")
#     plt.xlabel("Grade Level")
#     plt.ylabel("Score")
#     plt.title("Readability by Grade Level")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{DATASET_DIR}/ARI_FKGL_DC.png", dpi=400, bbox_inches="tight")
#     # plt.show()
#     plt.close()

#     plt.figure(figsize=(10, 5))
#     plt.plot(grades, words, marker="o", label="Words")
#     plt.xlabel("Grade Level")
#     plt.ylabel("Mean Words per Simplification")
#     plt.title("Grade vs. Mean Words per Simplification")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{DATASET_DIR}/grade_vs_words.png", dpi=400, bbox_inches="tight")
#     plt.close()
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(grades, sentences, marker="s", label="Sentences")
#     plt.xlabel("Grade Level")
#     plt.ylabel("Mean Sentences per Simplification")
#     plt.title("Grade vs. Mean Sentences per Simplification")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{DATASET_DIR}/grade_vs_sentences.png", dpi=400, bbox_inches="tight")
#     plt.close()
    
#     plt.figure(figsize=(10, 5))
#     plt.plot(grades, characters, marker="^", label="Characters")
#     plt.xlabel("Grade Level")
#     plt.ylabel("Mean Characters per Simplification")
#     plt.title("Grade vs. Mean Characters per Simplification")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(f"{DATASET_DIR}/grade_vs_characters.png", dpi=400, bbox_inches="tight")
#     plt.close()

# def plot_comparison(source_metrics, simplification_metrics):
#     # Unpacking the metrics
#     metrics = ['Words', 'Sentences', 'Characters']
#     source_values = [source_metrics['word_count'], source_metrics['sentence_count'], source_metrics['char_count']]
#     simplification_values = [simplification_metrics['word_count'], simplification_metrics['sentence_count'], simplification_metrics['char_count']]
    
#     # Plotting each metric separately
#     for i, metric in enumerate(metrics):
#         plt.figure(figsize=(10, 5))
#         plt.bar(['Source Text'], source_values[i], width=0.4, label='Source Text')
#         plt.bar(['Simplifications'], simplification_values[i], width=0.4, label='Simplifications')
#         plt.xlabel(metric)
#         plt.ylabel('Count')
#         plt.title(f'Comparison of {metric} Count')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f"{DATASET_DIR}/comparison_count_{metric.lower()}.png", dpi=400, bbox_inches="tight")
#         plt.close()
    
#     print("Plots saved successfully!")

# def compute_source_metrics(data):
#     source_fkgl_scores = [entry["source_metrics"]["FKGL"] for entry in data]
#     source_fre_scores = [entry["source_metrics"]["FRE"] for entry in data]
#     source_ari_scores = [entry["source_metrics"]["ARI"] for entry in data]
#     source_dale_chall_scores = [entry["source_metrics"]["Dale-Chall"] for entry in data]
#     source_word_count = [entry["source_metrics"]["word_count"] for entry in data]
#     source_char_count = [entry["source_metrics"]["char_count"] for entry in data]
#     source_sent_count = [entry["source_metrics"]["sentence_count"] for entry in data]

#     avg_metrics = {
#         "FKGL": sum(source_fkgl_scores) / len(source_fkgl_scores),
#         "FRE": sum(source_fre_scores) / len(source_fre_scores),
#         "ARI": sum(source_ari_scores) / len(source_ari_scores),
#         "Dale-Chall": sum(source_dale_chall_scores) / len(source_dale_chall_scores),
#         "char_count": sum(source_char_count) / len(source_char_count),
#         "word_count": sum(source_word_count) / len(source_word_count),
#         "sentence_count": sum(source_sent_count) / len(source_sent_count)
#     }

#     return avg_metrics

# def main():
#     dataset = load_jsonl(DATASET_PATH)
#     dataset = keep_eng(dataset)
#     stats = compute_statistics(dataset)

#     metrics_by_grade = compute_target_metrics_by_grade(dataset)
#     plot_target_metrics_by_grade(metrics_by_grade)

#     source_metrics = compute_source_metrics(dataset)
#     target_metrics_average = compute_target_metrics_average(dataset)
#     plot_comparison(source_metrics, target_metrics_average)

#     log_file = f"{DATASET_DIR}/log.txt"
#     with open(log_file, "w", encoding="utf-8") as log:
#         original_stdout = sys.stdout
#         sys.stdout = log
    
#         print(f"Dataset Statistics: {DATASET_NAME}")
#         for key, value in stats.items():
#             print(f"{key}: {value}")

#         print(f"\nAverage by grade:")
#         for key, value in metrics_by_grade.items():
#             print(f"{key}: {value}")

#         print(f"\nReadability of Source Texts:")
#         for key, value in source_metrics.items():
#             print(f"{key}: {value}")

#         sys.stdout = original_stdout
    
#     print(f"Log saved to {log_file}.")


# if __name__ == "__main__":
#     main()
