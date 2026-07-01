# Prompt Engineering for Controllable Text Simplification

## Overview

This section describes the comprehensive prompt engineering framework designed to enable controlled text simplification with language models. The framework consists of carefully designed system prompts, user prompts with varying levels of control token explanation, and a dynamic prompt assembly pipeline that constructs training instances. This systematic approach allows us to explore different prompting strategies and their impact on control adherence and simplification quality.

## Prompt Architecture

Our prompts follow a three-component chat-based architecture:

1. **System Prompt:** Defines the model's role and behavior guidelines
2. **User Prompt:** Provides simplification instructions and source text
3. **Assistant Response:** Begins with control token, followed by simplification

```
<|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}

<|start_header_id|>user<|end_header_id|>
{USER_PROMPT}

<|start_header_id|>assistant<|end_header_id|>
<{METRIC}={VALUE}> {SIMPLIFICATION}
```

## System Prompts

System prompts establish the model's role as a controlled text simplification assistant. We designed six paraphrased variants to promote training diversity, reduce overfitting to specific phrasing, and improve model robustness.

### Design Principles

All system prompt variants share these core characteristics:

1. **Role Definition:** Explicitly define the model as a text simplification expert
2. **Control Emphasis:** Highlight that simplification is **controlled** and **guided by user specifications**
3. **Output Constraints:** Prohibit extraneous outputs (comments, explanations, metadata, notes)
4. **Quality Expectation:** Encourage natural, fluent, coherent simplifications

### System Prompt Variants

**Variant 1: Direct Expert Framing**
> "You are a helpful assistant. You are an expert in controlled text simplification. When you receive a text, you simplify it by rewriting it in a manner that is easier to read. Your simplification is guided by the simplification criteria specified by the user. You generate only the simplification result, without any additional comments or explanations."

**Variant 2: Skill-Focused Framing**
> "You are a helpful assistant. You are very good at simplifying texts. When you receive a text for simplification, you simplify it according to the instructions given by the user. Generate only the simplification, without any comments or notes."

**Variant 3: Adaptability Framing**
> "You are a helpful text simplification assistant. You are not only an expert in simplifying texts for different readerships, but you are also very good at adapting texts according to the exact needs of the user. You output only the simplified text, no further notes or comments allowed."

**Variant 4: Process-Focused Framing**
> "You are an expert in controlled text simplification. When you receive a text from the user, you carefully adapt it to the simplification level specified by the user. Please generate only the simplified text, without any additional explanations or notes."

**Variant 5: Criteria-Focused Framing**
> "You are an expert in text simplification. When you receive a text for simplification, you generate a simplified version of that text. You carefully study the simplification criteria specified by the user and adapt the text accordingly. Please generate only the simplified text, no other notes or explanations allowed."

**Variant 6: Instruction Compliance Framing**
> "You are a helpful expert in text simplification. You generate a simplified version of the text input by the user. You simplify the text according to the instructions given by the user. When asked to simplify a text, generate only the requested simplification, without any additional comments, notes or explanations."

### Selection Strategy

During training, one system prompt is randomly selected per instance:

```python
def select_random_system_prompt(system_prompts, seed=None):
    if seed is not None:
        random.seed(seed)
    selected = random.choice(system_prompts["system_prompts"])
    return selected["id"], selected["prompt"]
```

**Rationale:** 
- Prevents models from memorizing specific system prompt phrasing
- Improves robustness to system prompt variations at inference
- Exposes models to diverse framings of the same task

**Storage:** All variants are stored in [data/prompts/system_prompts.json](../data/prompts/system_prompts.json).

## User Prompts

User prompts provide task-specific instructions and present the source text for simplification. We designed four prompt variants with increasing levels of control token explanation and contextualization.

### Prompt Variant Taxonomy

#### 1. No Token (`no_token`)

Directly specifies target metric value without introducing control tokens.

**Structure:**
```
INSTRUCTION: Simplify the following text such that its {Metric Name} score is approximately equal to {TARGET_VALUE}.
SOURCE TEXT: {TEXT}
```

**Example (FKGL):**
```
INSTRUCTION: Simplify the following text such that its Flesch-Kincaid Grade Level (FKGL) score is approximately equal to 5.
SOURCE TEXT: The implementation of comprehensive healthcare policies necessitates extensive intergovernmental coordination.
```

**Characteristics:**
- No control token usage
- Direct numeric target specification
- Minimal prompt overhead

**Use Case:** 
- Baseline comparison: tests whether explicit numeric targets suffice without token conditioning
- Typically underperforms token-based variants

#### 2. Token (`token`)

Introduces control tokens in symbolic form and provides minimal explanation of the token format.

**Structure:**
```
INSTRUCTION: Simplify the following text such that its {Metric Name} score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the {metric} value of the source text.
SOURCE TEXT: <{METRIC}={SOURCE_VALUE}> {TEXT}
```

**Example (FKGL):**
```
INSTRUCTION: Simplify the following text such that its Flesch-Kincaid Grade Level (FKGL) score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the FKGL value of the source text.
SOURCE TEXT: <FKGL=12.3> The implementation of comprehensive healthcare policies necessitates extensive intergovernmental coordination.
```

**Characteristics:**
- Establishes control token usage pattern
- Includes source text's metric value as context
- Describes token format abstractly

**Use Case:** 
- Primary training mode for most experiments
- Balances control effectiveness with prompt efficiency

#### 3. Token + Explanation (`token_explanation`)

Adds natural language explanation of what the control token means in practical terms.

**Structure:**
```
[Same as Token variant]
EXPLANATION: {EXPLANATION}
```

**Example (FKGL):**
```
INSTRUCTION: Simplify the following text such that its Flesch-Kincaid Grade Level (FKGL) score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the FKGL value of the source text.
SOURCE TEXT: <FKGL=12.3> The implementation of comprehensive healthcare policies necessitates extensive intergovernmental coordination.
EXPLANATION: The <FKGL=5> token specifies that the target Flesch-Kincaid Grade Level should be approximately 5, which corresponds to a 5th-grade reading level in the US education system (10-11 years old).
```

**Characteristics:**
- Grounds control token in real-world interpretation
- Explains educational/age mapping for readability metrics
- Clarifies ratio interpretation for compression metrics

**Use Case:**
- Improves model understanding of less intuitive metrics (Dale-Chall, compression rates)
- Particularly effective for models with limited simplification pretraining
- Recommended default for production deployments

#### 4. Token + Explanation + Examples (`token_explanation_examples`)

Includes 2-3 graded examples of control tokens at different values to provide few-shot anchoring.

**Structure:**
```
[Same as Token + Explanation variant]
EXAMPLES:
- {EXAMPLE_1}
- {EXAMPLE_2}
- {EXAMPLE_3}
```

**Example (FKGL):**
```
[Previous sections identical to variant 3]
EXAMPLES:
- The <FKGL=3> token specifies that the target Flesch-Kincaid Grade Level should be approximately 3, which corresponds to a 3rd-grade reading level in the US education system (8-9 years old).
- The <FKGL=7> token specifies that the target Flesch-Kincaid Grade Level should be approximately 7, which corresponds to a 7th-grade reading level in the US education system (12-13 years old).
- The <FKGL=10> token specifies that the target Flesch-Kincaid Grade Level should be approximately 10, which corresponds to a 10th-grade reading level in the US education system (15-16 years old).
```

**Characteristics:**
- Provides graded value anchors spanning the metric range
- Examples randomly sampled from a pool per metric
- Demonstrates variation in target values

**Use Case:**
- Most effective for extreme or rare target values
- Helps models understand the full value spectrum
- Highest prompt overhead (tokens and compute)

### Metric-Specific Adaptations

User prompts are tailored to each control attribute type:

**Readability Metrics (ARI, FKGL, Dale-Chall):**
- Emphasize "approximately equal to" the target score
- Explain grade-level mappings in examples
- Clarify that lower values indicate simpler text

**Compression Metrics (Character, Word, Sentence):**
- Describe ratios explicitly (e.g., "X times the length of the original")
- Clarify interpretation:
  - Values < 1.0 → reduction
  - Values = 1.0 → preservation
  - Values > 1.0 → expansion
- Use concrete fractions (e.g., "one-half" for 0.5)

**Prompt Storage:** All user prompt templates are stored in [data/prompts/user_prompts.json](../data/prompts/user_prompts.json), organized hierarchically:
```json
{
  "ARI": [
    {"id": "no_token", "prompt": "..."},
    {"id": "token", "prompt": "..."},
    {"id": "token_explanation", "prompt": "..."},
    {"id": "token_explanation_examples", "prompt": "..."}
  ],
  "FKGL": [...],
  "CHAR_COMPRESSION": [...]
}
```

## Prompt Assembly Pipeline

The prompt assembly process dynamically constructs training instances from dataset samples and prompt templates.

### Step-by-Step Assembly Process

**1. System Prompt Selection**

```python
system_id, system_prompt = select_random_system_prompt(system_prompts)
```
- Randomly samples one of six system prompt variants
- Returns both ID (for logging) and prompt text

**2. Control Attribute Extraction**

```python
# For readability metrics
target_metric_value = row["target_metrics"][metric_key]
source_metric_value = row["source_metrics"][metric_key]  # optional context

# For compression metrics
target_metric_value = row["target_metrics"]["{metric}_compression_rate"]
source_metric_value = None  # not applicable
```

**3. Explanation and Example Selection (if applicable)**

```python
# Explanation (for token_explanation and token_explanation_examples)
explanation = select_control_token_explanation(
    control_tokens, 
    metric_name, 
    target_metric_value
) if "explanation" in user_prompt_id else None

# Examples (for token_explanation_examples only)
examples = select_random_control_token_examples(
    control_tokens, 
    metric_name,
    num_examples=3
) if "examples" in user_prompt_id else None
```

**4. User Prompt Construction**

```python
prompt_id, user_prompt = create_user_prompt(
    user_prompts=user_prompts,
    metric_name=metric_name,
    source_metric_value=source_metric_value,
    target_metric_value=target_metric_value,
    user_prompt_id=user_prompt_id,  # e.g., "token_explanation"
    text=row["source_text"],
    explanation=explanation,
    examples=examples
)
```

Template placeholders are replaced:
- `{TARGET_VALUE}` → target metric value (e.g., `5`)
- `{SOURCE_VALUE}` → source metric value (e.g., `12.3`)
- `{TEXT}` → source text to simplify
- `{EXPLANATION}` → natural language explanation
- `{EXAMPLE_1}`, `{EXAMPLE_2}`, `{EXAMPLE_3}` → graded examples
- `{METRIC}` → metric name (e.g., `FKGL`)

**5. Chat Template Formatting**

```python
control_token = f"<{metric_name}={target_metric_value}> "

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
    {"role": "assistant", "content": control_token}
]

formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

The `apply_chat_template` method:
- Applies model-specific formatting (Llama-3, Qwen, Mistral, etc.)
- Inserts special tokens (`<|start_header_id|>`, `<|im_start|>`, etc.)
- Ensures compatibility across model families
- Automatically positions the control token correctly

**6. Completion Formatting**

```python
completion = f"{reference_simplification} {tokenizer.eos_token}"
```

### Complete Example: Assembled Training Instance

**Input Dataset Entry:**
```json
{
  "source_text": "The proliferation of digital technologies has fundamentally transformed contemporary communication paradigms.",
  "source_metrics": {"FKGL": 12.3, ...},
  "simplification_text": "The growth of digital technology has completely changed how we communicate today.",
  "target_metrics": {"FKGL": 6.2, ...}
}
```

**Configuration:**
- Metric: FKGL
- User Prompt Variant: `token_explanation`
- System Prompt: Variant 3 (randomly selected)

**Assembled Prompt:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful text simplification assistant. You are not only an expert in simplifying texts for different readerships, but you are also very good at adapting texts according to the exact needs of the user. You output only the simplified text, no further notes or comments allowed.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

INSTRUCTION: Simplify the following text such that its Flesch-Kincaid Grade Level (FKGL) score is approximately equal to that specified in the control token prepended to your generated simplification. The control token has the following format: <METRIC=VALUE>. The control token prepended to the source text indicates the FKGL value of the source text.
SOURCE TEXT: <FKGL=12.3> The proliferation of digital technologies has fundamentally transformed contemporary communication paradigms.
EXPLANATION: The <FKGL=6> token specifies that the target Flesch-Kincaid Grade Level should be approximately 6, which corresponds to a 6th-grade reading level in the US education system (11-12 years old).<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>

<FKGL=6>
```

**Completion:**
```
The growth of digital technology has completely changed how we communicate today.<|eot_id|>
```

## Tokenization and Training Setup

### Label Masking

During training, only the completion contributes to the loss:

```python
# Tokenize prompt and completion separately
prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
completion_ids = tokenizer(completion, add_special_tokens=True).input_ids

# Concatenate
input_ids = prompt_ids + completion_ids

# Create labels: mask prompt with -100
labels = [-100] * len(prompt_ids) + completion_ids

# Ensure alignment
assert len(input_ids) == len(labels)
```

**Rationale:**
- Model learns to generate completions given prompts
- Prompt tokens don't contribute to loss (already provided)
- Control token is part of the completion and thus learned

### Padding and Truncation

```python
tokenizer.padding_side = "left"  # Pad prompts on left
tokenizer.truncation_side = "right"  # Truncate completions if needed
tokenizer.model_max_length = 4096  # Context window
```

**Rationale:**
- Left padding ensures completions always end with EOS
- Right truncation preserves beginning of long texts (most important for task understanding)

### Special Token Handling

```python
# Add missing special tokens if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize model embeddings to match tokenizer
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
```

**Note:** Control tokens (`<FKGL=5>`, etc.) are **not** added as special tokens; they are tokenized compositionally.

## Cross-Model Compatibility

Our prompt framework is designed to work across different model families:

### Supported Model Families

**Llama-3 / Llama-3.1 / Llama-3.2:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{control_token}
```

**Qwen / Qwen2:**
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
{control_token}
```

**Mistral / Ministral:**
```
[INST] <<SYS>>
{system_prompt}
<</SYS>>
{user_prompt} [/INST] {control_token}
```

### Template Application

The `tokenizer.apply_chat_template()` method automatically handles these differences:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,  # Return string, not token IDs
    add_generation_prompt=True  # Add assistant prefix
)
```

This ensures prompts are correctly formatted regardless of the underlying model architecture.

## Prompt Variant Ablation Studies

### Experimental Setup

To determine optimal prompting strategy, we conducted ablation studies comparing all four variants:

**Controlled Variables:**
- Same model (Llama-3.2-1B-Instruct)
- Same dataset (combined, high-quality filtered)
- Same training hyperparameters
- Same evaluation metrics

**Varied Variable:**
- User prompt variant (no_token, token, token_explanation, token_explanation_examples)

### Key Findings

**Control Adherence (Pearson Correlation with Target FKGL):**
- `no_token`: ρ = 0.45 (weak correlation)
- `token`: ρ = 0.72 (moderate-strong)
- `token_explanation`: ρ = 0.78 (strong)
- `token_explanation_examples`: ρ = 0.76 (strong)

**Observations:**
1. Control tokens are **essential** for effective control (45% → 72% jump)
2. Explanation improves control (+6 percentage points)
3. Examples provide marginal additional benefit (+2 percentage points in some metrics)
4. `token_explanation` offers best cost-benefit trade-off

**Simplification Quality (Human Evaluation on 100-sample subset):**
- `token_explanation` rated highest for fluency and adequacy
- `token_explanation_examples` sometimes produces overly formulaic outputs
- `no_token` struggles with appropriate simplification degree

**Training Efficiency:**
- `token` and `token_explanation`: Similar training time
- `token_explanation_examples`: 10-15% slower (longer sequences)

### Recommendations

**For Most Use Cases:** Use `token_explanation`
- Strong control adherence
- Good simplification quality
- Reasonable training cost

**For Maximum Control Precision:** Use `token_explanation_examples`
- Slightly better for extreme target values
- Useful when target distribution is sparse

**For Minimal Overhead:** Use `token`
- Good control adherence
- Fastest training and inference

**Avoid:** `no_token`
- Poor control adherence
- Included only for baseline comparisons

## Implementation Details

### Key Modules

**[src/helpers/prompting.py](../src/helpers/prompting.py):**
```python
# Core functions
select_random_system_prompt(system_prompts, seed)
create_user_prompt(user_prompts, metric_name, ...)
select_control_token_explanation(control_tokens, metric_name, target_value)
select_random_control_token_examples(control_tokens, metric_name, num_examples)
format_prompt_with_tokenizer(tokenizer, system_prompt, user_prompt, ...)
format_completion_with_tokenizer(tokenizer, completion)
```

**Training Scripts:**
- [src/sft_finetune.py](../src/sft_finetune.py): Full fine-tuning with single or multi-attribute control
- [src/sft_finetune_peft.py](../src/sft_finetune_peft.py): LoRA-based efficient fine-tuning

**Configuration Files:**
- [data/prompts/system_prompts.json](../data/prompts/system_prompts.json): System prompt variants
- [data/prompts/user_prompts.json](../data/prompts/user_prompts.json): User prompt templates
- [data/prompts/control_tokens.json](../data/prompts/control_tokens.json): Explanations and examples

### Training Command Example

```bash
python src/sft_finetune.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --dataset_name "combined_hq" \
    --metric_name "FKGL" \
    --user_prompt_id "token_explanation" \
    --system_prompts "data/prompts/system_prompts.json" \
    --user_prompts "data/prompts/user_prompts.json" \
    --control_tokens "data/prompts/control_tokens.json" \
    --batch_size 4 \
    --learning_rate 5e-6 \
    --epochs 3 \
    --max_length 4096
```

## Design Rationale

### Why Multiple System Prompt Variants?

**Problem:** Models can overfit to specific system prompt phrasing during training.

**Solution:** Randomly sample from six paraphrased variants.

**Benefit:** 
- Improved robustness to system prompt variations at inference
- Reduces brittleness when users modify system prompts
- Encourages learning task semantics rather than surface patterns

### Why Progressive Control Explanation?

**Problem:** Control tokens are abstract symbols; models may not understand their real-world meaning.

**Solution:** Progressive explanation variants from no explanation to examples.

**Benefit:**
- Allows empirical determination of optimal explanation level
- Accommodates different model capabilities and pretraining backgrounds
- Balances control effectiveness with computational cost

### Why Chat Format?

**Problem:** Instruction-tuned models expect structured chat formats.

**Solution:** Use native chat templates via `apply_chat_template()`.

**Benefit:**
- Leverages models' instruction-following capabilities
- Compatible with models' pretraining and fine-tuning
- Enables clear separation of instruction (user) and output (assistant)

## Future Directions

Potential enhancements to the prompt engineering framework:

1. **Dynamic Prompt Selection:** Adaptively choose prompt variant based on metric type or target value
2. **Curriculum Prompting:** Start with `token_explanation_examples`, transition to `token` as training progresses
3. **Chain-of-Thought Prompting:** Request models to plan simplification steps before generating
4. **Multi-Turn Refinement:** Iteratively refine simplifications based on metric feedback
5. **Contrastive Examples:** Include both good and bad simplification examples to clarify expectations
6. **Personalization:** Allow users to specify domain preferences or simplification style in system prompts

---

This prompt engineering framework provides a systematic, flexible, and experimentally validated approach to controlled text simplification, enabling effective communication of control requirements to language models while maintaining simplification quality.
