# WikiLarge Dataset Processing History

## Overview

This document tracks the complete processing pipeline for the WikiLarge dataset in this repository, clarifying the naming conventions, downsampling strategies, and split creation process. **This is essential reading to avoid confusion about what "splitwise" and "global" actually mean in the context of WikiLarge.**

---

## Original WikiLarge Dataset (Zhang & Lapata, 2017)

### Source Information
- **Full name:** WikiLarge sentence simplification dataset
- **Publication:** Zhang & Lapata (2017)
- **Original files:** Pre-split into train/valid/test with `.src` and `.dst` files
  - `wiki.full.aner.ori.train.src` / `wiki.full.aner.ori.train.dst`
  - `wiki.full.aner.ori.valid.src` / `wiki.full.aner.ori.valid.dst`
  - `wiki.full.aner.ori.test.src` / `wiki.full.aner.ori.test.dst`

### Original Split Sizes
- **Training set:** ~296,400 sentence pairs (1 reference per source)
- **Validation set:** ~2,000 sentence pairs (1 reference per source)
- **Test set:** ~359 sentence pairs (1 reference per source in .dst file)
- **Total:** ~298,759 sentence pairs

### Key Characteristics
- **Domain:** General (Wikipedia → Simple Wikipedia)
- **Alignment:** Automatic sentence-level alignment
- **References:** Test set has 8 reference simplifications per source (stored separately for evaluation)
- **Quality:** Lower than human-annotated datasets due to automatic alignment

### ⚠️ Important: Multiple Reference Handling
The WikiLarge dataset structure uses:
- **`.src` files:** One line per source sentence
- **`.dst` files:** One line per source (the PRIMARY reference simplification)
- **Multiple references:** The 8 test references are stored in SEPARATE files (typically `test.ref0` through `test.ref7`)

**How this repository handles it:**
1. **Data loading ([Dataset.py](../src/classes/Dataset.py)):** Only loads `.src` and `.dst` files, creating 1:1 pairs
2. **Result:** Each source sentence gets exactly **1 simplification** in the dataset
3. **The 8 references:** Are NOT loaded or used in training/validation - only available for evaluation metrics
4. **Impact:** Training and internal evaluation use only the single `.dst` reference per source

---

## Processing in This Repository

### ⚠️ CRITICAL: Understanding the Two-Stage Process

WikiLarge processing happened in **TWO DISTINCT STAGES**, not one:

#### **Stage 1: Initial Loading and Global Downsampling** (UNDOCUMENTED SIZE)
The original 300k dataset was loaded via `WikiLarge` class in [src/classes/Dataset.py](../src/classes/Dataset.py):
- Loaded all three original splits (train/valid/test)
- Combined into single DataFrame with `original_split` column
- Converted to JSONL format with metadata preservation
- **THEN: Unknown intermediate downsampling occurred**, reducing from ~300k to ~2,000 instances
  - This was likely done via global stratified sampling (ignoring split boundaries)
  - The exact parameters of this first downsampling are not documented in code
  - Result: `wikilarge_ori/dataset.jsonl` with ~2,000 instances

#### **Stage 2: New Split Creation** (DOCUMENTED IN CODE)
From the ~2,000 instance subset, NEW splits were created via [src/generate_splits.py](../src/generate_splits.py):
- **Input:** Pre-downsampled ~2,000 instances (no longer has meaningful train/valid/test distinction)
- **Process:** 
  1. Remove outliers (FKGL, ARI, Dale-Chall, char_length at 1st/99th percentiles)
  2. Apply stratified sampling using FKGL with 25 bins
  3. Create NEW 80/10/10 splits (not preserving original splits)
- **Output:** `splits_new/wikilarge_ori_splitwise/` with:
  - Train: 1,473 instances (80%)
  - Val: 183 instances (10%)
  - Test: 199 instances (10%)
  - Total: 1,855 instances (after outlier removal)

---

## Understanding "Splitwise" vs "Global" Naming

### ⚠️ WARNING: Misleading Terminology

The terms "splitwise" and "global" in this repository **DO NOT** refer to the original WikiLarge splits!

#### What "Splitwise" ACTUALLY Means
**In the context of WikiLarge:** Creates NEW stratified 80/10/10 splits from an already-downsampled dataset.

**Common misconception:** That it preserves the original train/valid/test boundaries from Zhang & Lapata (2017).

**Reality:** By the time "splitwise" processing happens, the original splits are already destroyed through global downsampling in Stage 1.

#### What "Global" ACTUALLY Means
**In the context of WikiLarge:** Creates NEW stratified 80/10/10 splits without any consideration of previous split boundaries.

**Functionally identical to "splitwise"** for WikiLarge because both operate on the same pre-downsampled ~2k dataset.

#### Why This Naming Exists
The "splitwise" function `stratified_sampling_from_split()` in [src/generate_wikilarge_subset.py](../src/generate_wikilarge_subset.py) CAN preserve splits when:
1. The input data still has meaningful `original_split` metadata
2. Each original split is large enough to sample from

**This works for:** Med-EASi, Newsela (before downsampling)

**This CANNOT work for WikiLarge because:**
- Original test set (359 instances) is too small to sample 10% of 7,000 = 700 instances
- The data was already globally downsampled in Stage 1, destroying split boundaries

---

## File Locations and Sizes

### Dataset Files
```
data/datasets/wikilarge_ori/
├── dataset.jsonl                    [MISSING/DELETED - was ~2,000 instances]
└── stats/

data/splits_new/wikilarge_ori_splitwise/
├── train.jsonl                      [1,473 instances]
├── val.jsonl                        [183 instances]
├── test.jsonl                       [199 instances]
└── log.txt                          [Processing log showing outlier removal]

data/splits_flattened/wikilarge_ori_splitwise/
├── train.jsonl                      [1,473 instances - flattened format]
├── val.jsonl                        [183 instances - flattened format]
└── test.jsonl                       [199 instances - flattened format]

data/splits_flattened_filtered/wikilarge_ori_splitwise/
├── train.jsonl                      [Size varies - after quality filtering]
├── val.jsonl                        [Size varies - after quality filtering]
└── test.jsonl                       [Size varies - after quality filtering]
```

### Processing Scripts
- **Initial loading:** `src/classes/Dataset.py` → `WikiLarge` class
- **Format conversion:** `src/dataset_to_jsonl.py`
- **Split generation:** `src/generate_splits.py` (creates splits_new/)
- **Flattening:** `src/flatten_splits.py` (creates splits_flattened/)
- **Quality filtering:** `src/filter_flattened_splits.py` (creates splits_flattened_filtered/)
- **Subset creation experiments:** `src/generate_wikilarge_subset.py`, `src/explore_wikilarge_sampling.py`

---

## The Missing Piece: Stage 1 Downsampling

### What We Know
- Original dataset: ~300k instances
- Dataset after Stage 1: ~2,000 instances
- Reduction ratio: ~99.3% removed

### What We DON'T Know (Not in Code)
1. **When** was Stage 1 downsampling performed?
2. **What script** performed it?
3. **What stratification metric** was used?
4. **What bin size** was used?
5. **What random seed** was used?
6. **Were outliers removed** before sampling?

### Likely Scenario
Based on the code patterns, Stage 1 probably:
1. Loaded full 300k dataset
2. Applied global stratified sampling (ignoring `original_split`)
3. Used FKGL or similar readability metric for stratification
4. Sampled ~2,000 instances
5. Saved to `data/datasets/wikilarge_ori/dataset.jsonl`
6. **This file may have been deleted or was never committed to version control**

---

## Experiments and Divergence Analysis

### Purpose
Files like `explore_wikilarge_sampling.py` and `plot_wikilarge_divergencies.py` compare different sampling strategies.

### What They Actually Test
**They compare:** Different stratification metrics (FKGL, ARI, char_count, etc.) for creating representative subsets.

**They DO NOT compare:** Preserving original splits vs. creating new splits (both methods create new splits from the same ~2k base).

### Divergence Metrics
- **JSD (Jensen-Shannon Divergence):** Measures distributional similarity
- **EMD (Earth Mover's Distance):** Measures cost of transforming one distribution to another
- **KS (Kolmogorov-Smirnov):** Statistical test for distribution similarity

### Experiment Results Location
```
experiments/sample_from_wikilarge_ori/
└── num_bins_15/
    ├── all_divergence_results.json
    └── plots/
```

---

## Recommendations for Future Work

### If You Need Original WikiLarge Splits
1. **Re-download** the original Zhang & Lapata (2017) dataset
2. Load via `WikiLarge` class in `src/classes/Dataset.py`
3. **Do NOT downsample** before creating splits
4. Use only the test set (359 instances) for evaluation
5. Use original train set (~296k) for training
6. **For proper evaluation:** Load the 8 test references (test.ref0-7) separately for computing metrics like BLEU/SARI with multiple references

### If You Need a Representative Subset
1. Start with full 300k dataset
2. Apply **global** stratified sampling (ignore splits) to desired size
3. Create NEW 80/10/10 splits from the subset
4. **Do NOT call this "splitwise"** - it's misleading

### If You Want to Preserve Original Splits
1. This is **not possible** for WikiLarge at subset sizes > 2,000
2. Original test set (359) is too small for meaningful subsampling
3. Consider using original full splits instead

---

## Quality Filtering

After split creation, quality filtering is applied to remove low-quality simplifications.

### Filtering Criteria (Conjunctive)
Simplifications are retained only if ALL readability metrics DECREASE from source to target:
- Flesch-Kincaid Grade Level (FKGL) ↓
- Automated Readability Index (ARI) ↓
- Dale-Chall Readability Score ↓

### Rationale
True simplifications should reduce reading difficulty. Instances where complexity increases likely represent:
- Automatic alignment errors
- Poor quality simplifications
- Edge cases

### Impact
WikiLarge (automatic alignment) typically has 20-40% removal rate, higher than human-annotated datasets.

---

## Data Schema Evolution

### Stage 1: JSONL with Simplifications Array
```json
{
  "global_id": "wikilarge_ori_000001",
  "source_text": "...",
  "source_metrics": {...},
  "metadata": {
    "dataset": "wikilarge_ori",
    "original_split": "train",  // Preserved from original files
    "domain": "general",
    ...
  },
  "simplifications": [
    {
      "simplification_text": "...",  // From .dst file ONLY
      "target_metrics": {...},
      "simplification_version": 1,  // Always 1 for WikiLarge
      ...
    }
    // NOTE: Array has exactly 1 element for WikiLarge
    // The 8 test references are NOT loaded
  ]
}
```

### Stage 2: Flattened Format
```json
{
  "global_id": "wikilarge_ori_000001",
  "source_text": "...",
  "source_metrics": {...},
  "metadata": {
    "dataset": "wikilarge_ori",
    "experiment_split": "",  // NEW splits don't preserve original
    "domain": "general",
    ...
  },
  "simplification_text": "...",  // Flattened from array
  "target_metrics": {...},
  ...
}
```

Note: `original_split` is removed during split generation (`generate_splits.py` line 75-76).

---

## Summary: What Actually Happened to WikiLarge

1. ✅ **Original dataset loaded:** 300k instances from Zhang & Lapata (2017) files
2. ❓ **Stage 1 downsampling:** 300k → ~2k (UNDOCUMENTED, possibly global stratified sampling)
3. ✅ **Outlier removal:** ~2k → 1,855 (removing extreme FKGL, ARI, Dale-Chall, char_length)
4. ✅ **NEW split creation:** 1,855 → Train(1,473) + Val(183) + Test(199)
5. ✅ **Flattening:** Convert from simplifications array to one instance per simplification
6. ✅ **Quality filtering:** Remove instances where readability metrics don't decrease

### Key Insight
**The "wikilarge_ori_splitwise" dataset does NOT preserve original WikiLarge splits.** Both "splitwise" and "global" strategies create NEW splits from a pre-downsampled subset.

---

## Frequently Asked Questions

### Q: Why not use the original test set of 359 instances?
A: You can and should if you want to compare to published results on WikiLarge. However, this project creates uniform evaluation sets across datasets.

### Q: Can I recover the original splits?
A: Not from files in this repository. You need to re-download the original WikiLarge dataset.

### Q: What does "ori" mean in "wikilarge_ori"?
A: "Original" - distinguishing this from other WikiLarge variants. However, it's misleading since it's actually a heavily downsampled version.

### Q: Why is the test set so small in the original WikiLarge?
A: Zhang & Lapata (2017) created a small but high-quality test set with 8 references per source for more reliable evaluation. However, only 1 reference (from the .dst file) is used in this repository.

### Q: Where are the 8 test references and why aren't they used?
A: The 8 references are stored in separate files (test.ref0-7) that are not loaded by this repository's data processing pipeline. The `WikiLarge` class in `Dataset.py` only loads the `.src` and `.dst` files, treating WikiLarge as a simple 1:1 pairing dataset. To use all 8 references, you would need to:
1. Modify the dataset loader to read additional reference files
2. Store multiple simplifications per source in the `simplifications` array
3. Update the flattening logic to handle multiple references appropriately

### Q: Should I use "splitwise" or "global" sampling for WikiLarge?
A: For WikiLarge specifically, they're functionally equivalent. The distinction only matters for datasets where original splits are still meaningful (Med-EASi, Newsela).

---

## References

- Zhang, X., & Lapata, M. (2017). Sentence Simplification with Deep Reinforcement Learning. *EMNLP 2017*.
- Original WikiLarge: Derived from aligning English Wikipedia with Simple English Wikipedia
- Repository location: `/storage/homefs/hh25g551/ondemand/data/taming-CATS/`

---

**Last Updated:** January 22, 2026  
**Created By:** Documentation of existing pipeline  
**Purpose:** Prevent future confusion about WikiLarge processing and split preservation
