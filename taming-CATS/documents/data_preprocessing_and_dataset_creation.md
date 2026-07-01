# Data Preprocessing and Dataset Creation

## Overview

This section describes the comprehensive data preprocessing pipeline and dataset creation methodology employed for controllable automatic domain-specific text simplification. The pipeline processes multiple publicly available simplification datasets, standardizes their formats, computes readability metrics, and generates quality-filtered training splits suitable for supervised fine-tuning of language models.

## Source Datasets

Our work integrates four distinct text simplification datasets covering different domains and simplification approaches:

### Newsela
**Domain:** News  
**Alignment Level:** Document-level  
**Annotation:** Human-annotated  
**Language:** English  
**Description:** The Newsela corpus consists of news articles manually simplified to multiple grade levels (0-4), providing explicit readability targets. Each article exists in multiple versions of decreasing complexity, making it ideal for studying controlled simplification across specific reading difficulty levels.

### Med-EASi
**Domain:** Medical  
**Alignment Level:** Sentence-level  
**Annotation:** Human-annotated  
**Language:** English  
**Description:** Med-EASi (Medical Easy-to-read Adapted Simplification) contains sentence-aligned pairs of medical expert language and their corresponding simplified versions. This dataset addresses the critical need for accessible medical information, featuring simplifications that reduce technical terminology while preserving semantic accuracy.

### WikiLarge
**Domain:** General/Wikipedia  
**Alignment Level:** Sentence-level  
**Annotation:** Automatic  
**Language:** English  
**Description:** WikiLarge is a large-scale automatically-aligned corpus derived from Wikipedia's Simple English edition. Despite being automatically generated, it provides extensive training data for general-domain simplification. Due to its size (283,040 sentence pairs in the original dataset), we apply stratified sampling to create a representative subset for computational efficiency.

### SimPA
**Domain:** Public Administration  
**Alignment Level:** Sentence-level  
**Annotation:** Human-annotated with explicit simplification dimensions  
**Language:** English  
**Description:** SimPA comprises two specialized subsets targeting distinct linguistic simplification dimensions:
- **SimPA-Lexical:** Focuses on vocabulary simplification (replacing complex words with simpler synonyms)
- **SimPA-Syntactic:** Targets syntactic simplification (sentence restructuring, splitting, etc.)

To avoid redundancy while maximizing coverage, we merge these subsets by: (1) retaining all unique source sentences from each subset, and (2) for overlapping source sentences, randomly assigning them to either the lexical or syntactic subset with 50-50 probability. This approach ensures no duplicate source texts while preserving the diversity of both simplification types.

## Data Preprocessing Pipeline

### Stage 1: Format Standardization and Metric Computation

All source datasets are converted to a unified JSON Lines (JSONL) schema that enables consistent downstream processing. Each entry in the standardized format contains:

#### Global Metadata
- **global_id:** Unique identifier per source text (`{dataset_name}_{counter:06d}`)
- **source_text:** Original complex text
- **metadata:** 
  - Dataset name, domain, language
  - Annotation type (human/automatic)
  - Alignment level (sentence/document)
  - Original split information (train/valid/test, where applicable)

#### Source Metrics
For each source text, we compute eight readability and structural metrics:
- **Readability indices:** Flesch Reading Ease (FRE), Automated Readability Index (ARI), Flesch-Kincaid Grade Level (FKGL), Dale-Chall Score
- **Structural features:** Character count, alphanumeric character count, word count, sentence count

#### Simplifications Array
Each source text has one or more associated simplifications, each containing:
- **simplification_text:** The simplified version
- **simplification_version:** Version number (for multi-level simplifications like Newsela)
- **grade_level:** Target reading grade (Newsela only, -1 for others)
- **simplification_dimensions:** Linguistic categories (lexical/syntactic flags for SimPA)
- **target_metrics:** All readability and structural metrics computed for the simplified text
- **Compression rates:** Character, word, and sentence compression ratios relative to source
- **Similarity metrics:** BLEU and BERTScore computed between source and simplified texts

This standardization is implemented through dataset-specific loader classes (Newsela, MedEASi, WikiLarge, SimPALex, SimPASyn) that handle the unique formatting of each source corpus, followed by unified metric computation using the `Metrics` class, which leverages the textstat library for readability scores and the evaluate library for BERTScore calculations.

### Stage 2: Data Splitting and Stratification

To ensure representative train/validation/test splits across varying text complexities, we employ stratified sampling based on readability metrics:

#### Outlier Removal
Before splitting, we remove extreme outliers that could skew model training:
- Filter texts falling below the 1st or above the 99th percentile for: FKGL, ARI, Dale-Chall, and character length
- This removes approximately 2% of data points while preserving the core distribution

#### Stratified Splitting
1. **Stratification metric:** We use FKGL (Flesch-Kincaid Grade Level) as the primary stratification dimension, selected based on its interpretability and strong correlation with text complexity
2. **Binning:** The FKGL distribution is divided into 25 equal-width bins
3. **Split ratios:** From each bin, we sample: 80% training, 10% validation, 10% test
4. **Randomization:** A fixed seed (42) ensures reproducibility

This stratification ensures balanced complexity distributions across all splits, preventing models from being exposed to drastically different difficulty levels during training versus evaluation.

#### Sampling for Large Datasets
For datasets exceeding 3,000 source texts (particularly Newsela and WikiLarge), we apply random sampling post-filtering to maintain computational feasibility while preserving distributional characteristics.

### Stage 3: Flattening to Training Instances

The hierarchical JSONL format (one source text with multiple simplifications) is flattened to create individual training instances:

#### Flattening Process
Each simplification becomes a standalone training example containing:
- Source text and its metrics
- Single simplification text and its metrics
- Metadata (dataset, domain, linguistic dimensions)
- Compression rates and similarity scores

#### Selective Sampling
For computational efficiency, we optionally retain a configurable fraction (e.g., 40%) of flattened instances through random sampling with a fixed seed. This is particularly useful for datasets with high simplification multiplicities (e.g., Newsela with 5 versions per article).

### Stage 4: Quality Filtering

To ensure training data quality, we apply metric-based filtering that enforces expected simplification properties:

#### Filtering Criteria
We retain only instances where **all** of the following readability metrics decrease from source to target:
- Flesch-Kincaid Grade Level (FKGL)
- Automated Readability Index (ARI)
- Dale-Chall Readability Score

**Rationale:** True simplifications should reduce reading difficulty across multiple metrics. Instances where complexity increases or remains unchanged likely represent annotation errors, misalignments, or cases where "simplification" was spurious.

#### Filtering Statistics
The filtering step creates a "high-quality" (HQ) variant of each split, with removal statistics logged per dataset and metric. Typical removal rates range from 10-30% depending on dataset quality, with automatically-aligned corpora (WikiLarge) showing higher removal rates than human-annotated datasets (Newsela, Med-EASi, SimPA).

Venn diagrams are generated to visualize overlaps in instances failing different metrics, revealing that most removed instances fail multiple criteria simultaneously, validating the conjunctive filtering approach.

### Stage 5: Multi-Dataset Combination

To create a unified training corpus encompassing diverse domains and simplification types, we combine the filtered splits:

#### Combined Dataset Composition
- **Med-EASi** (medical domain)
- **SimPA** (public administration, lexical + syntactic)
- **WikiLarge** (general domain)

**Note:** Newsela is available for combination but was selectively included/excluded in different experimental configurations due to its document-level alignment and multiple versions per source.

#### Combination Procedure
1. Concatenate train, validation, and test splits separately across selected datasets
2. Shuffle each split to intermix examples from different domains and datasets
3. Preserve split boundaries (no data leakage between train/validation/test)

This combined dataset enables models to learn domain-agnostic simplification strategies while being exposed to diverse linguistic phenomena.

## Dataset Schema

The final training instances follow this flattened schema:

```json
{
  "global_id": "string",
  "source_text": "string",
  "source_metrics": {
    "FRE": "float",
    "ARI": "float", 
    "FKGL": "float",
    "Dale-Chall": "float",
    "char_count": "int",
    "alphanum_count": "int",
    "word_count": "int",
    "sentence_count": "int"
  },
  "metadata": {
    "dataset": "string",
    "domain": "string",
    "language": "string",
    "annotation": "string",
    "level": "string"
  },
  "simplification_text": "string",
  "target_metrics": {
    "FRE": "float",
    "ARI": "float",
    "FKGL": "float", 
    "Dale-Chall": "float",
    "char_count": "int",
    "alphanum_count": "int",
    "word_count": "int",
    "sentence_count": "int",
    "char_compression_rate": "float",
    "word_compression_rate": "float",
    "sentence_compression_rate": "float",
    "BLEU": "float",
    "BERTScore": "float"
  },
  "simplification_version": "int",
  "grade_level": "int",
  "simplification_dimensions": {
    "linguistic": {
      "lexical": "boolean",
      "syntactic": "boolean"
    }
  }
}
```

## Implementation Details

### Key Scripts and Modules
- **dataset_to_jsonl.py:** Loads raw datasets and converts to standardized JSONL with metric computation
- **generate_splits.py:** Performs outlier removal, stratified sampling, and train/val/test splitting
- **flatten_splits.py:** Converts hierarchical format to flat training instances with optional subsampling
- **filter_flattened_splits.py:** Applies quality filters based on readability metric reduction
- **create_combined_dataset.py:** Merges multiple filtered datasets into unified corpus
- **Classes:**
  - **Dataset.py:** Contains loader classes (Newsela, MedEASi, WikiLarge, SimPALex, SimPASyn)
  - **Metrics.py:** Computes all readability metrics, similarity scores, and compression rates

### Computational Considerations
- **Metric computation:** BERTScore uses the `microsoft/deberta-xlarge-mnli` model for semantic similarity
- **Batch processing:** Large datasets processed iteratively to manage memory
- **Reproducibility:** All random operations use fixed seeds (default: 42)

## Quality Assurance

Multiple validation steps ensure data quality:
1. **Duplicate detection:** SimPA subset merging explicitly handles overlapping sources
2. **Metric validation:** All readability scores validated for reasonable ranges
3. **Length sanity checks:** Extreme text lengths filtered during outlier removal
4. **Distribution monitoring:** Histograms and statistical summaries generated for each split to verify stratification effectiveness
5. **Manual inspection:** Random samples inspected to verify alignment quality and simplification appropriateness

## Data Availability

All processed dataset splits are version-controlled and uploaded to Hugging Face Hub for reproducibility and community access. The standardized format enables straightforward integration with transformer-based training pipelines (Hugging Face `datasets` library).

---

This comprehensive preprocessing pipeline ensures that our controllable simplification models are trained on high-quality, diverse, and appropriately stratified data that spans multiple domains and simplification dimensions. The multi-stage approach—from format standardization through quality filtering to multi-dataset combination—provides a robust foundation for investigating controllable text simplification across domain-specific contexts.
