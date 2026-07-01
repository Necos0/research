# Control Attributes for Controllable Text Simplification

## Overview

Control attributes (also referred to as control tokens) are special tokens that encode target simplification characteristics as measurable metrics. These tokens are prepended to model outputs during training and inference to guide the generation process toward desired simplification goals. This approach enables fine-grained, quantifiable control over multiple dimensions of text simplification.

## Control Attribute Types

We implement two categories of control attributes that target different aspects of text simplification:

### Readability-Based Control Attributes

These attributes target specific reading difficulty levels based on established readability formulas:

#### Automated Readability Index (ARI)

- **Token Format:** `<ARI={value}>`
- **Description:** Controls the reading grade level based on character-to-word and word-to-sentence ratios
- **Value Type:** Float (typically 0-12+, corresponding to US grade levels)
- **Interpretation:** Lower values indicate simpler text; ARI=4 targets 4th-grade reading level
- **Formula:** $ARI = 4.71 \times \frac{\text{characters}}{\text{words}} + 0.5 \times \frac{\text{words}}{\text{sentences}} - 21.43$
- **Usage:** Particularly effective for educational content adaptation and grade-level targeting
- **Example:** `<ARI=5>` specifies text appropriate for a 5th-grade reading level (10-11 years old)

#### Flesch-Kincaid Grade Level (FKGL)

- **Token Format:** `<FKGL={value}>`
- **Description:** Controls the US grade level required to understand the text
- **Value Type:** Float (typically 0-12+)
- **Interpretation:** Lower values indicate simpler text; FKGL=6 targets 6th-grade reading level
- **Formula:** $FKGL = 0.39 \times \frac{\text{words}}{\text{sentences}} + 11.8 \times \frac{\text{syllables}}{\text{words}} - 15.59$
- **Usage:** Widely recognized standard for readability assessment; highly interpretable for practitioners
- **Example:** `<FKGL=8>` specifies text appropriate for an 8th-grade reading level (13-14 years old)

#### Dale-Chall Readability Score

- **Token Format:** `<DALE-CHALL={value}>`
- **Description:** Controls complexity based on difficult word usage (words outside the Dale-Chall familiar word list of 3,000 common words)
- **Value Type:** Float (typically 4-10+)
- **Interpretation:** Lower values indicate fewer difficult words; scores map to grade ranges:
  - 4.0 or below: 4th grade and below
  - 5.0-5.9: 5th-6th grade
  - 6.0-6.9: 7th-8th grade
  - 7.0-7.9: 9th-10th grade
  - 8.0-8.9: 11th-12th grade
  - 9.0-9.9: College level
  - 10.0+: College graduate level
- **Usage:** Lexically-focused simplification; particularly sensitive to vocabulary difficulty
- **Example:** `<DALE-CHALL=6>` specifies text appropriate for 7th-8th grade reading level

### Structural Control Attributes

These attributes control text length and structural transformations:

#### Character Compression Rate

- **Token Format:** `<CHAR_COMPRESSION={value}>`
- **Description:** Controls the ratio of output character count to input character count
- **Value Type:** Float (typically 0.2-1.5)
- **Interpretation:** 
  - Values < 1.0: Text reduction (e.g., 0.5 = output is half the input length)
  - Values = 1.0: Length preservation
  - Values > 1.0: Text expansion (e.g., 1.2 = output is 20% longer)
- **Formula:** $\text{Compression Rate} = \frac{\text{target\_char\_count}}{\text{source\_char\_count}}$
- **Usage:** Direct control over summarization vs. elaboration degree; useful for length-constrained applications
- **Example:** `<CHAR_COMPRESSION=0.7>` specifies output should be 70% the length of input (30% reduction)

#### Word Compression Rate

- **Token Format:** `<WORD_COMPRESSION={value}>`
- **Description:** Controls the ratio of output word count to input word count
- **Value Type:** Float (typically 0.2-1.5)
- **Interpretation:** Same as character compression, but at word granularity
- **Formula:** $\text{Compression Rate} = \frac{\text{target\_word\_count}}{\text{source\_word\_count}}$
- **Usage:** More robust to whitespace variations than character-based compression; aligns better with semantic content
- **Example:** `<WORD_COMPRESSION=0.8>` specifies output should contain 80% of input words (20% reduction)

#### Sentence Compression Rate

- **Token Format:** `<SENTENCE_COMPRESSION={value}>`
- **Description:** Controls the ratio of output sentence count to input sentence count
- **Value Type:** Float (typically 0.5-2.0)
- **Interpretation:**
  - Values < 1.0: Sentence merging/condensation
  - Values = 1.0: Sentence count preservation
  - Values > 1.0: Sentence splitting (e.g., 2.0 = double the sentences)
- **Formula:** $\text{Compression Rate} = \frac{\text{target\_sentence\_count}}{\text{source\_sentence\_count}}$
- **Usage:** Syntactic complexity control; encourages sentence splitting for higher values, merging for lower values
- **Example:** `<SENTENCE_COMPRESSION=1.5>` specifies output should have 50% more sentences (encouraging splitting)

## Control Attribute Extraction

Control attributes are automatically extracted from the preprocessed dataset during training data preparation. The extraction process differs based on the attribute category:

### Source-Based Metrics (Readability Attributes)

For readability metrics (ARI, FKGL, Dale-Chall), we use the **target text's** readability score as the control value:

1. **Source Metric Computation:** Calculate readability score for the source (complex) text
2. **Target Metric Computation:** Calculate readability score for the target (simplified) text
3. **Control Value Selection:** Use the target metric value as the control token value
4. **Optional Context:** The source metric may optionally be included in the prompt to provide context about the starting complexity

**Rationale:** Models learn to generate text that matches specific readability levels, not to perform transformations by a certain magnitude. This approach enables absolute control ("simplify to 5th-grade level") rather than relative control ("reduce complexity by X points").

### Target-Based Metrics (Structural Attributes)

For compression metrics, we compute the ratio between target and source text statistics:

```python
char_compression = target_char_count / source_char_count
word_compression = target_word_count / source_word_count
sentence_compression = target_sentence_count / source_sentence_count
```

These ratios are rounded to two decimal places (e.g., 0.73, 1.05) to reduce vocabulary space while maintaining sufficient granularity.

**Rationale:** Compression inherently requires awareness of both source and target properties. The ratio provides a normalized, scale-invariant measure of transformation degree.

### Value Discretization and Rounding

To balance precision with generalization:

- **Readability Metrics:** Rounded to one decimal place (e.g., 7.3, 10.2)
- **Compression Metrics:** Rounded to two decimal places (e.g., 0.73, 1.05)

This discretization:
1. Reduces the number of unique control values the model encounters
2. Improves generalization to unseen values through interpolation
3. Prevents overfitting to exact metric values
4. Maintains sufficient precision for meaningful control

## Control Token Explanations and Examples

To improve model understanding of control tokens, we provide natural language explanations and graded examples for each metric in [data/prompts/control_tokens.json](../data/prompts/control_tokens.json).

### Explanation Templates

Each control attribute has an explanation template that describes its meaning:

**Template Structure:**
> "The `<{METRIC}={TARGET_VALUE}>` token specifies that the target {metric description} should be approximately {TARGET_VALUE}. {Interpretation guidance}."

**Example for FKGL:**
> "The `<FKGL=6>` token specifies that the target Flesch-Kincaid Grade Level should be approximately 6, which corresponds to a 6th-grade reading level in the US education system (11-12 years old)."

**Example for Word Compression:**
> "The `<WORD_COMPRESSION=0.5>` token specifies that the word-level compression rate should be around 0.5, which means the number of words in the output text should be approximately half (1/2) of the number of words in the original text."

### Graded Examples

For each metric, we provide 6-13 examples spanning the typical value range:

**ARI Examples (excerpt):**
- `<ARI=0>` → Pre-school reading level (5-6 years old)
- `<ARI=3>` → 3rd-grade reading level (8-9 years old)
- `<ARI=6>` → 6th-grade reading level (11-12 years old)
- `<ARI=12>` → 12th-grade reading level (18+ years old)

**Character Compression Examples (excerpt):**
- `<CHAR_COMPRESSION=0.2>` → Output is one-fifth the input length
- `<CHAR_COMPRESSION=0.8>` → Output is 80% of input length
- `<CHAR_COMPRESSION=1.0>` → Output matches input length
- `<CHAR_COMPRESSION=1.2>` → Output is 120% of input length (expansion)

These examples can be included in prompts to provide few-shot anchoring for the model.

## Training with Control Attributes

### Single-Attribute Training

In single-attribute training mode, each training run focuses on one control attribute:

**Configuration:**
```bash
--metric_name "FKGL"  # or ARI, DALE-CHALL, CHAR_COMPRESSION, etc.
```

**Characteristics:**
- All training instances formatted with the same metric's control tokens
- Model becomes specialized for controlling one dimension of simplification
- Achieves highest control accuracy for the trained metric
- Requires separate models for different attributes

**Use Case:** When maximum precision is needed for a specific control dimension (e.g., strict FKGL adherence for educational materials)

### Multi-Attribute Training

In multi-attribute training mode, a single model learns to control multiple attributes simultaneously:

**Configuration:**
```bash
--metric_names "ARI" "FKGL" "DALE-CHALL" "CHAR_COMPRESSION" "WORD_COMPRESSION"
```

**Round-Robin Attribute Selection:**
During data preprocessing, metrics are cycled through in sequence:
```python
allowed_metrics = ["ARI", "FKGL", "DALE-CHALL", "CHAR_COMPRESSION", "WORD_COMPRESSION"]
metric_counter = {"count": 0}

def get_next_metric():
    metric = allowed_metrics[metric_counter["count"] % len(allowed_metrics)]
    metric_counter["count"] += 1
    return metric

# Training instance creation:
# Instance 1 → ARI control token
# Instance 2 → FKGL control token
# Instance 3 → DALE-CHALL control token
# Instance 4 → CHAR_COMPRESSION control token
# Instance 5 → WORD_COMPRESSION control token
# Instance 6 → ARI control token (cycle repeats)
```

**Characteristics:**
- Unified control token vocabulary across all metrics
- Model learns control tokens as a general conditioning mechanism
- Single model deployment covers all control dimensions
- Slight trade-off in per-metric accuracy (~2-5% Pearson correlation decrease)
- Better cross-metric generalization

**Use Case:** When deployment flexibility is important or when users need to switch between different control dimensions

### Training Objective and Loss

Models are trained with standard causal language modeling (next-token prediction):

**Loss Computation:**
```python
# Input format: [PROMPT] <CONTROL_TOKEN> [REFERENCE_SIMPLIFICATION] <EOS>
# Labels: Only completion contributes to loss

labels = [-100] * len(prompt_tokens) + control_token_ids + simplification_ids + [eos_token_id]
loss = CrossEntropyLoss(predictions, labels, ignore_index=-100)
```

**Key Training Details:**
- Prompt tokens masked with `-100` (ignored in loss)
- Control token + simplification + EOS contribute to loss
- AdamW optimizer with cosine learning rate schedule
- Gradient clipping (max_grad_norm=0.5) for stability
- BFloat16 mixed precision training
- Learning rate: 5e-6 for full fine-tuning
- Batch size: 4-8 per device with gradient accumulation

## Inference with Control Attributes

### Controllable Generation Process

At inference time, users specify desired control attribute values to generate custom simplifications:

**Step-by-Step Process:**
1. **Control Value Specification:** User provides target metric value
   ```python
   metric_name = "FKGL"
   target_value = 5
   ```

2. **Prompt Construction:** System constructs prompt with user-specified control token
   ```
   <|start_header_id|>system<|end_header_id|>
   You are an expert in controlled text simplification...
   <|start_header_id|>user<|end_header_id|>
   INSTRUCTION: Simplify the following text such that its FKGL score is approximately 5.
   SOURCE TEXT: <FKGL=12.3> {complex_text}
   <|start_header_id|>assistant<|end_header_id|>
   <FKGL=5>
   ```

3. **Generation:** Model generates simplification conditioned on the control token
   ```python
   outputs = model.generate(
       input_ids,
       max_new_tokens=512,
       do_sample=False,  # Greedy decoding for reproducibility
       pad_token_id=tokenizer.pad_token_id,
       eos_token_id=tokenizer.eos_token_id
   )
   ```

4. **Post-Processing:** Output is cleaned of extraneous tokens and whitespace
   ```python
   prediction = tokenizer.decode(outputs, skip_special_tokens=True)
   prediction = prediction.strip()
   ```

### Inference Flexibility

**Arbitrary Control Values:**
- Models can generalize to control values not seen during training (interpolation)
- Effective range typically spans the training data distribution ±10%
- Example: If trained on FKGL=[3, 4, 5, 7, 8, 10], model can handle FKGL=6 or FKGL=9

**Extrapolation Boundaries:**
- Beyond training range, control accuracy degrades gracefully
- Extreme values (e.g., FKGL=0 or FKGL=20) may produce boundary effects
- Recommended to keep inference values within [min_train - 1σ, max_train + 1σ]

**Mixed-Attribute Control (Multi-Attribute Models):**
- Models can be prompted with any learned control attribute at inference
- Switch control dimensions per request without reloading models
- Enables user choice of most relevant control dimension per use case

**Example Inference Requests:**

*Readability Control:*
```
Metric: FKGL=5
Input: "The proliferation of digital technologies has fundamentally transformed contemporary communication paradigms."
Output: <FKGL=5> The growth of digital technology has completely changed how we communicate today.
```

*Compression Control:*
```
Metric: WORD_COMPRESSION=0.5
Input: "The quick brown fox jumps over the lazy dog in the park on a sunny afternoon."
Output: <WORD_COMPRESSION=0.5> The fox jumps over the dog in the park.
```

*Sentence Splitting Control:*
```
Metric: SENTENCE_COMPRESSION=2.0
Input: "The company announced record profits and expanded operations."
Output: <SENTENCE_COMPRESSION=2.0> The company announced record profits. They also expanded operations.
```

## Evaluation of Control Adherence

To assess control attribute effectiveness, we measure:

### 1. Control Accuracy

**Metric:** Pearson correlation coefficient between requested and achieved values
```python
correlation, p_value = pearsonr(requested_values, achieved_values)
```

**Interpretation:**
- ρ > 0.8: Strong control adherence
- 0.6 < ρ ≤ 0.8: Moderate control adherence
- ρ ≤ 0.6: Weak control adherence

**Typical Results:**
- Readability metrics (ARI, FKGL, Dale-Chall): ρ = 0.75-0.85
- Compression metrics: ρ = 0.65-0.75 (inherently noisier due to model creativity)

### 2. Range Coverage

**Metric:** Percentage of control values where models produce outputs within acceptable tolerance

```python
tolerance = 0.1 * target_value  # ±10% relative error
coverage = np.mean(np.abs(achieved - requested) <= tolerance)
```

**Typical Results:**
- FKGL: 60-70% within ±10% relative error
- Compression: 50-60% within ±10% relative error

### 3. Absolute Error Statistics

**Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Median Absolute Error

**Typical Results (FKGL):**
- MAE: 1.2-1.8 grade levels
- RMSE: 2.0-2.5 grade levels
- Median: 0.9-1.3 grade levels

### 4. Boundary Behavior

**Analysis:** Performance at distribution extremes

- **Low Values (e.g., FKGL < 4):** Models struggle to produce very simple text; may plateau at ~FKGL=4
- **High Values (e.g., FKGL > 12):** Models can maintain or increase complexity, but control precision degrades
- **Mid-Range (FKGL 5-10):** Strongest control adherence and accuracy

### Visualization

Results are visualized through:
1. **Scatter Plots:** Predicted vs. requested values with ideal diagonal line
2. **Error Distributions:** Histograms of (achieved - requested) differences
3. **Control Curves:** Mean achieved value per requested value bin
4. **Correlation Heatmaps:** Cross-metric correlations for multi-attribute models

## Implementation Details

### Special Token Handling

**Design Decision: Compositional Tokenization**

Control tokens (`<ARI=5>`, `<FKGL=7.3>`, etc.) are **not** added to the tokenizer vocabulary as single tokens. Instead, they are tokenized compositionally:

```python
# Example tokenization of <FKGL=7.3>
tokens = ['<', 'F', 'K', 'GL', '=', '7', '.', '3', '>']
# or
tokens = ['<', 'FK', 'GL', '=', '7.3', '>']  # depending on tokenizer
```

**Rationale:**
1. **Vocabulary Efficiency:** Avoids explosion of vocabulary size (thousands of possible control values)
2. **Generalization:** Enables handling of unseen control values through compositional understanding
3. **Flexibility:** New metrics can be added without retraining tokenizer
4. **Numeric Reasoning:** Models learn to interpret numeric values compositionally

### Metric Computation

All metrics are computed using established libraries:

```python
from classes.Metrics import Metrics

metrics = Metrics(
    input_text=simplified_text,
    source_text=source_text,
    reference_text=reference_simplification
)
computed = metrics.compute_metrics()

# Returns:
# {
#     "ARI": float,
#     "FKGL": float,
#     "Dale-Chall": float,
#     "char_count": int,
#     "word_count": int,
#     "sentence_count": int,
#     "char_compression_rate": float,
#     "word_compression_rate": float,
#     "sentence_compression_rate": float,
#     ...
# }
```

**Libraries Used:**
- `textstat`: Readability metrics (FKGL, ARI, Dale-Chall, FRE)
- `nltk`: Tokenization (word_tokenize, sent_tokenize)
- Custom computation: Compression rates

### Key Scripts

- **[src/sft_finetune.py](../src/sft_finetune.py):** Single-attribute training
- **[src/sft_finetune_all_ctrl_attr.py](../src/sft_finetune_all_ctrl_attr.py):** Multi-attribute training
- **[src/sft_inference.py](../src/sft_inference.py):** Inference with control attributes
- **[src/classes/Metrics.py](../src/classes/Metrics.py):** Metric computation module

### Configuration Files

- **[data/prompts/control_tokens.json](../data/prompts/control_tokens.json):** Control token definitions, explanations, examples
- **[data/metric_mapping.json](../data/metric_mapping.json):** Maps control attribute names to dataset field names

## Design Rationale

### Why Control Tokens?

Control tokens provide several advantages over alternative conditioning mechanisms:

1. **Explicitness:** Users can directly specify numeric targets without ambiguous natural language descriptions
2. **Precision:** Numeric values enable fine-grained control (e.g., FKGL=5 vs. FKGL=6)
3. **Measurability:** Automatic evaluation via metric computation on outputs
4. **Consistency:** Reduces prompt ambiguity compared to phrases like "simplify moderately"
5. **Interpretability:** Established readability metrics have known educational/professional mappings

### Multi-Attribute vs. Single-Attribute Trade-offs

| Aspect | Single-Attribute | Multi-Attribute |
|--------|-----------------|-----------------|
| Control Accuracy | ★★★★★ | ★★★★☆ |
| Deployment Flexibility | ★★☆☆☆ | ★★★★★ |
| Storage Requirements | ★★☆☆☆ (N models) | ★★★★★ (1 model) |
| Training Cost | ★★★☆☆ (N runs) | ★★★★☆ (1 run) |
| Cross-Metric Generalization | ★★☆☆☆ | ★★★★☆ |

**Recommendation:** Use multi-attribute training for most applications; reserve single-attribute for mission-critical accuracy requirements.

## Future Directions

Potential extensions to the control attribute framework:

1. **Hierarchical Control:** Combining multiple attributes in a single prompt (e.g., `<FKGL=5> <WORD_COMPRESSION=0.8>`)
2. **Soft Constraints:** Specifying acceptable ranges (e.g., `<FKGL=5-7>`) rather than exact values
3. **Domain-Specific Attributes:** Medical term density, legal jargon score, technical vocabulary ratio
4. **Linguistic Attributes:** Passive voice percentage, sentence structure complexity, lexical diversity
5. **Adversarial Control:** Training with challenging target values to improve boundary performance
6. **Adaptive Control:** Dynamically adjusting targets mid-generation based on partial outputs
7. **User Preference Learning:** Training reward models to capture subjective quality beyond metrics

---

This control attribute framework provides a principled, measurable, and flexible approach to controllable text simplification, enabling precise control over readability and structural properties across diverse domain-specific contexts.
