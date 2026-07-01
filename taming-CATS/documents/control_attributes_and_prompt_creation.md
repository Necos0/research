# Control Attribute Extraction and Prompt Creation

## Overview

This section describes the methodology for controllable text simplification through control attributes (also referred to as control tokens) and the comprehensive prompt engineering framework that enables fine-grained control over simplification outputs. Our approach allows models to generate simplifications targeting specific readability levels or structural properties by conditioning generation on explicit control signals embedded in the input prompts.

## Control Attributes

Control attributes are special tokens that encode target simplification characteristics as measurable metrics. These tokens are prepended to model outputs during training and inference to guide the generation process toward desired simplification goals.

### Supported Control Attribute Types

We implement two categories of control attributes:

#### 1. Readability-Based Control Attributes

These attributes target specific reading difficulty levels based on established readability formulas:

**Automated Readability Index (ARI)**
- **Token Format:** `<ARI={value}>`
- **Description:** Controls the reading grade level based on character-to-word and word-to-sentence ratios
- **Value Type:** Float (typically 0-12+, corresponding to US grade levels)
- **Interpretation:** Lower values indicate simpler text; ARI=4 targets 4th-grade reading level
- **Usage:** Particularly effective for educational content adaptation and grade-level targeting

**Flesch-Kincaid Grade Level (FKGL)**
- **Token Format:** `<FKGL={value}>`
- **Description:** Controls the US grade level required to understand the text
- **Value Type:** Float (typically 0-12+)
- **Interpretation:** Lower values indicate simpler text; FKGL=6 targets 6th-grade reading level
- **Usage:** Widely recognized standard for readability assessment; highly interpretable for practitioners

**Dale-Chall Readability Score**
- **Token Format:** `<DALE-CHALL={value}>`
- **Description:** Controls complexity based on difficult word usage (words outside the Dale-Chall familiar word list)
- **Value Type:** Float (typically 4-10+)
- **Interpretation:** Lower values indicate fewer difficult words; scores map to grade ranges (e.g., 5-6 = 5th-6th grade)
- **Usage:** Lexically-focused simplification; particularly sensitive to vocabulary difficulty

#### 2. Structural Control Attributes

These attributes control text length and structural transformations:

**Character Compression Rate**
- **Token Format:** `<CHAR_COMPRESSION={value}>`
- **Description:** Controls the ratio of output character count to input character count
- **Value Type:** Float (typically 0.2-1.5)
- **Interpretation:** 
  - Values < 1.0: Text reduction (e.g., 0.5 = output is half the input length)
  - Values = 1.0: Length preservation
  - Values > 1.0: Text expansion (e.g., 1.2 = output is 20% longer)
- **Usage:** Direct control over summarization vs. elaboration degree

**Word Compression Rate**
- **Token Format:** `<WORD_COMPRESSION={value}>`
- **Description:** Controls the ratio of output word count to input word count
- **Value Type:** Float (typically 0.2-1.5)
- **Interpretation:** Same as character compression, but at word granularity
- **Usage:** More robust to whitespace variations than character-based compression

**Sentence Compression Rate**
- **Token Format:** `<SENTENCE_COMPRESSION={value}>`
- **Description:** Controls the ratio of output sentence count to input sentence count
- **Value Type:** Float (typically 0.5-2.0)
- **Interpretation:**
  - Values < 1.0: Sentence merging/condensation
  - Values = 1.0: Sentence count preservation
  - Values > 1.0: Sentence splitting (e.g., 2.0 = double the sentences)
- **Usage:** Syntactic complexity control; encourages sentence splitting for lower values

### Control Attribute Extraction

Control attributes are automatically extracted from the preprocessed dataset during training data preparation:

1. **Source-Based Metrics (Readability):** For readability metrics (ARI, FKGL, Dale-Chall), we use the **target text's** readability score as the control value. The source text's readability may optionally be included in the prompt for context.

2. **Target-Based Metrics (Compression):** For compression metrics, we compute the ratio between target and source text statistics:
   - Character compression = `target_char_count / source_char_count`
   - Word compression = `target_word_count / source_word_count`
   - Sentence compression = `target_sentence_count / source_sentence_count`

3. **Value Discretization:** While raw metric values are used during training, they are typically rounded to one decimal place for readability metrics and two decimal places for compression rates to reduce the vocabulary space and improve generalization.

### Control Token Explanations and Examples

To improve model understanding of control tokens, we provide natural language explanations and graded examples for each metric:

**Explanation Template:**
> "The `<{METRIC}={TARGET_VALUE}>` token specifies that the target {metric description} should be approximately {TARGET_VALUE}. {Interpretation guidance}."

**Example for ARI:**
> "The `<ARI=5>` token specifies that the target Automated Readability Index (ARI) score should be approximately 5, which corresponds to a 5th-grade reading level in the US education system (10-11 years old)."

These explanations are stored in [data/prompts/control_tokens.json](../data/prompts/control_tokens.json) and can be optionally included in prompts during training (see Prompt Variants below).

## Prompt Engineering Framework

Our prompt engineering framework enables flexible, systematic exploration of different prompting strategies for controlled simplification. The framework consists of three components: system prompts, user prompts, and control token integration.

### System Prompts

System prompts establish the model's role as a text simplification assistant. We created six paraphrased variants to promote training diversity and reduce overfitting to specific phrasing:

**Core Characteristics:**
- Define the model as an expert in **controlled** text simplification
- Emphasize adherence to user-specified simplification criteria
- Explicitly prohibit extraneous outputs (comments, explanations, metadata)
- Encourage natural, fluent simplifications

**Example System Prompt (Variant 1):**
> "You are a helpful assistant. You are an expert in controlled text simplification. When you receive a text, you simplify it by rewriting it in a manner that is easier to read. Your simplification is guided by the simplification criteria specified by the user. You generate only the simplification result, without any additional comments or explanations."

**Example System Prompt (Variant 3):**
> "You are a helpful text simplification assistant. You are not only an expert in simplifying texts for different readerships, but you are also very good at adapting texts according to the exact needs of the user. You output only the simplified text, no further notes or comments allowed."

During training, one system prompt is randomly selected per instance to increase robustness. All variants are stored in [data/prompts/system_prompts.json](../data/prompts/system_prompts.json).

### User Prompts

User prompts provide simplification instructions and present the source text. We designed multiple prompt variants with increasing levels of control token explanation and contextualization.

#### Prompt Variants by Control Explicitness

**1. No Token (`no_token`)**
- Directly specifies target metric value without control tokens
- **Format:**
  ```
  INSTRUCTION: Simplify the following text such that its {Metric Name} score is approximately equal to {TARGET_VALUE}.
  SOURCE TEXT: {TEXT}
  ```
- **Use Case:** Baseline comparison; tests whether explicit numeric targets suffice without token conditioning

**2. Token (`token`)**
- Introduces control tokens in symbolic form
- Includes source text metric value as context
- **Format:**
  ```
  INSTRUCTION: Simplify the following text such that its {Metric Name} score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the {metric} value of the source text.
  SOURCE TEXT: <{METRIC}={SOURCE_VALUE}> {TEXT}
  ```
- **Use Case:** Primary training mode; establishes control token usage pattern

**3. Token + Explanation (`token_explanation`)**
- Adds natural language explanation of the control token's meaning
- **Format:**
  ```
  [Same as Token variant]
  EXPLANATION: {EXPLANATION}
  ```
- **Use Case:** Enhances model understanding of token semantics; particularly useful for complex metrics like Dale-Chall

**4. Token + Explanation + Examples (`token_explanation_examples`)**
- Includes 2-3 examples of control tokens at different values
- **Format:**
  ```
  [Same as Token + Explanation variant]
  EXAMPLES:
  - {EXAMPLE_1}
  - {EXAMPLE_2}
  - {EXAMPLE_3}
  ```
- **Use Case:** Few-shot learning within prompts; provides graded examples to anchor value ranges

#### Metric-Specific Adaptations

User prompts are tailored to each metric type:

- **Readability Metrics (ARI, FKGL, Dale-Chall):** Emphasize "approximately equal to" the target score
- **Compression Metrics:** Describe ratios explicitly (e.g., "output should be X times the length of the original")

All prompt templates are stored in [data/prompts/user_prompts.json](../data/prompts/user_prompts.json), organized by metric with each variant accessible by its ID.

### Prompt Assembly Pipeline

The prompt assembly process dynamically constructs training instances:

1. **System Prompt Selection:** Randomly sample one of six system prompt variants
2. **Control Attribute Extraction:** Retrieve target metric value from dataset instance
3. **User Prompt Construction:**
   - Select appropriate prompt template based on metric and variant
   - Substitute placeholders:
     - `{TARGET_VALUE}`: Target metric value
     - `{SOURCE_VALUE}`: Source metric value (if applicable)
     - `{TEXT}`: Source text to simplify
     - `{EXPLANATION}`: Control token explanation (if using `token_explanation` or higher)
     - `{EXAMPLE_1}`, `{EXAMPLE_2}`, `{EXAMPLE_3}`: Randomly sampled examples (if using `token_explanation_examples`)
4. **Chat Template Formatting:** Apply model-specific chat template using the tokenizer's `apply_chat_template` method
5. **Control Token Prepending:** Append control token (`<{METRIC}={VALUE}>`) to the assistant's response prefix
6. **Completion Formatting:** Concatenate reference simplification with EOS token

**Example Assembled Training Instance (FKGL, token_explanation variant):**

**Prompt:**
```
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You are an expert in controlled text simplification...

<|start_header_id|>user<|end_header_id|>
INSTRUCTION: Simplify the following text such that its Flesch-Kincaid Grade Level (FKGL) score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the FKGL value of the source text.
SOURCE TEXT: <FKGL=12.3> The implementation of comprehensive healthcare policies necessitates extensive intergovernmental coordination and substantial fiscal appropriations.
EXPLANATION: The <FKGL=6> token specifies that the target Flesch-Kincaid Grade Level should be approximately 6, which corresponds to a 6th-grade reading level in the US education system (11-12 years old).

<|start_header_id|>assistant<|end_header_id|>
<FKGL=6>
```

**Completion:**
```
Setting up full healthcare plans requires governments to work together and spend a lot of money. <|eot_id|>
```

## Training with Control Attributes

### Single-Attribute Training

In single-attribute training mode, each training run focuses on one control attribute:

- **Configuration:** `--metric_name {METRIC}` specifies the target metric (e.g., `ARI`, `FKGL`, `CHAR_COMPRESSION`)
- **Data Processing:** All training instances are formatted with prompts and control tokens for the specified metric
- **Model Specialization:** Results in a model specialized for controlling one dimension of simplification

### Multi-Attribute Training

In multi-attribute training mode, a single model learns to control multiple attributes:

- **Configuration:** `--metric_names {METRIC_1} {METRIC_2} ... {METRIC_N}` specifies multiple metrics
- **Round-Robin Attribute Selection:** During data preprocessing, metrics are cycled through in sequence:
  - Instance 1 → Metric 1
  - Instance 2 → Metric 2
  - ...
  - Instance N → Metric N
  - Instance N+1 → Metric 1 (cycle repeats)
- **Unified Control Token Space:** The model learns a single vocabulary of diverse control tokens
- **Generalization Benefit:** Encourages the model to understand control tokens as a general conditioning mechanism rather than memorizing specific patterns

**Implementation:**
```python
allowed_metrics = ["ARI", "FKGL", "DALE-CHALL", "CHAR_COMPRESSION", "WORD_COMPRESSION"]
metric_counter = {"count": 0}

def get_next_metric():
    metric = allowed_metrics[metric_counter["count"] % len(allowed_metrics)]
    metric_counter["count"] += 1
    return metric
```

### Training Objective

Models are trained with standard causal language modeling (next-token prediction):

- **Input:** Tokenized prompt with control token
- **Labels:** Reference simplification with control token and EOS token
- **Masking:** Prompt tokens are masked with `-100` in labels; only the completion (control token + simplification + EOS) contributes to loss
- **Optimization:** AdamW optimizer with cosine learning rate schedule and gradient clipping

## Inference with Control Attributes

### Controllable Generation Process

At inference time, users can specify desired control attribute values to generate custom simplifications:

1. **Control Value Specification:** User provides target metric value (e.g., `ARI=5`, `WORD_COMPRESSION=0.7`)
2. **Prompt Construction:** System constructs prompt with the user-specified control token
3. **Generation:** Model generates simplification conditioned on the control token
4. **Post-Processing:** Output is cleaned of extraneous tokens and whitespace

### Inference Flexibility

**Arbitrary Control Values:**
- Models can generalize to control values not seen during training (interpolation)
- Effective range typically spans the training data distribution ±10%

**Mixed-Attribute Control (Multi-Attribute Models):**
- Multi-attribute trained models can be prompted with any learned control attribute at inference
- Enables users to choose the most relevant dimension (readability vs. compression) per use case

**Example Inference Request:**
```
User: Simplify to 5th-grade reading level (FKGL=5)
Input: "The proliferation of digital technologies has fundamentally transformed contemporary communication paradigms."
```

**Model Output:**
```
<FKGL=5> The growth of digital technology has completely changed how we communicate today.
```

## Evaluation of Control Adherence

To assess control attribute effectiveness, we measure:

1. **Control Accuracy:** Pearson correlation between requested and achieved metric values
2. **Range Coverage:** Percentage of control values where models produce outputs within ±10% of target
3. **Boundary Behavior:** Performance at extreme values (e.g., very low or very high grade levels)

Results are logged per-metric and visualized through scatter plots (predicted vs. requested values) and error distributions.

## Implementation Details

### Key Modules and Scripts

**Prompt Construction Helpers ([src/helpers/prompting.py](../src/helpers/prompting.py)):**
- `select_random_system_prompt()`: Random system prompt sampling
- `create_user_prompt()`: Template-based user prompt construction with placeholder substitution
- `select_control_token_explanation()`: Retrieves explanations from control_tokens.json
- `select_random_control_token_examples()`: Samples N examples for few-shot prompting
- `format_prompt_with_tokenizer()`: Applies model-specific chat templates
- `format_completion_with_tokenizer()`: Formats completions with EOS tokens

**Training Scripts:**
- **[src/sft_finetune.py](../src/sft_finetune.py):** Single-attribute training
- **[src/sft_finetune_all_ctrl_attr.py](../src/sft_finetune_all_ctrl_attr.py):** Multi-attribute training with round-robin metric selection

**Inference Scripts:**
- **[src/sft_inference.py](../src/sft_inference.py):** Generates simplifications with specified control attributes; computes metrics on outputs

**Configuration Files:**
- **[data/prompts/control_tokens.json](../data/prompts/control_tokens.json):** Control token definitions, explanations, examples
- **[data/prompts/system_prompts.json](../data/prompts/system_prompts.json):** System prompt variants
- **[data/prompts/user_prompts.json](../data/prompts/user_prompts.json):** User prompt templates per metric and variant
- **[data/metric_mapping.json](../data/metric_mapping.json):** Maps control attribute names to dataset field names

### Tokenizer and Model Considerations

**Special Token Handling:**
- Custom control tokens (`<ARI=...>`, `<FKGL=...>`, etc.) are **not** added to the tokenizer vocabulary
- Instead, they are tokenized compositionally (e.g., `<`, `ARI`, `=`, `5`, `>` as separate tokens)
- This avoids vocabulary explosion and enables generalization to unseen control values

**Padding and Truncation:**
- Left padding during training (completion always ends with EOS)
- Truncation at configurable max length (default 4096 tokens)

**Chat Template Usage:**
- Models use native chat templates (e.g., Llama-3's `<|start_header_id|>...<|end_header_id|>` format)
- `tokenizer.apply_chat_template()` ensures compatibility across model families (Llama, Qwen, Mistral)

## Design Rationale and Ablations

### Why Control Tokens?

Control tokens provide several advantages over alternative conditioning mechanisms:

1. **Explicitness:** Users can directly specify targets without ambiguous natural language descriptions
2. **Precision:** Numeric values enable fine-grained control (e.g., FKGL=5 vs. FKGL=6)
3. **Measurability:** Automatic evaluation via metric computation on outputs
4. **Consistency:** Reduces prompt ambiguity compared to phrases like "simplify moderately"

### Prompt Variant Rationale

- **No Token:** Establishes whether numeric targets alone suffice (answer: no, models struggle without token conditioning)
- **Token:** Core training mode; most computationally efficient
- **Token + Explanation:** Improves understanding of less intuitive metrics (Dale-Chall, compression rates)
- **Token + Examples:** Provides graded anchors; most effective for extreme or rare target values

Empirically, `token_explanation` offers the best trade-off between performance and context length.

### Multi-Attribute vs. Single-Attribute Training

**Single-Attribute:**
- **Pros:** Maximum specialization; slightly better control accuracy for the trained metric
- **Cons:** Requires separate models per metric; higher storage and deployment costs

**Multi-Attribute:**
- **Pros:** Unified model; better generalization across metrics; supports flexible inference
- **Cons:** Slightly lower per-metric accuracy (~2-5% Pearson correlation drop)

For most applications, multi-attribute training is preferred due to its flexibility and cost-efficiency.

## Future Directions

Potential extensions to the control attribute framework include:

1. **Hierarchical Control:** Combining multiple attributes in a single prompt (e.g., `<FKGL=5> <WORD_COMPRESSION=0.8>`)
2. **Soft Constraints:** Specifying acceptable ranges rather than exact values (e.g., `<FKGL=5-7>`)
3. **Domain-Specific Attributes:** Incorporating domain-aware metrics (e.g., medical term density, legal jargon score)
4. **User Preference Learning:** Training reward models to capture subjective simplification quality beyond metrics
5. **Adaptive Control:** Dynamically adjusting control values during generation based on partial outputs

---

This control attribute and prompt engineering framework provides a principled, flexible, and measurable approach to controllable text simplification, enabling domain-specific adaptation and fine-grained control over multiple dimensions of simplification quality.
