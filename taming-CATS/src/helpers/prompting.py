import random


def select_random_system_prompt(system_prompts, seed=None):
    """Randomly selects a system prompt from the list and returns its ID and text."""
    if seed is not None:
        random.seed(seed)
    selected_prompt = random.choice(system_prompts["system_prompts"])  # Seed set for reproducibility
    return selected_prompt["id"], selected_prompt["prompt"]

def select_random_user_prompt(user_prompts, metric_name, source_metric_value, target_metric_value, user_prompt_id, seed=None):
    """Randomly selects a user prompt from the list and returns its ID and text."""
    if seed is not None:
        random.seed(seed)
    selected_prompts = user_prompts.get(metric_name)
    if selected_prompts:
        selected_prompt = random.choice(selected_prompts[user_prompt_id])
        selected_prompt = selected_prompt["prompt"].replace("{TARGET_VALUE}", str(target_metric_value))
        selected_prompt = selected_prompt["prompt"].replace("{SOURCE_VALUE}", str(source_metric_value))
        return selected_prompt["id"], selected_prompt
    return None, None

def select_control_token_explanation(control_tokens, metric_name, target_metric_value):
    """Selects a control token explanation based on the metric name and value."""
    control_token = control_tokens.get(metric_name)
    if control_token:
        return control_token["explanation"].replace("{TARGET_VALUE}", str(target_metric_value))
    return None

def select_random_control_token_examples(control_tokens, metric_name, num_examples=2, seed=None):
    """Selects a random set of control token examples based on the metric name and value."""
    if seed is not None:
        random.seed(seed)
    control_token = control_tokens.get(metric_name)
    if control_token and "examples" in control_token:
        # examples = random.sample(control_token["examples"], num_examples)
        examples = random.sample(control_token["examples"], min(num_examples, len(control_token["examples"])))
        return examples
    return []

def create_user_prompt(user_prompts, metric_name, source_metric_value, target_metric_value, user_prompt_id, text, explanation=None, examples=None):
    """Constructs a user prompt based on the user prompt type."""
    # Access the list of prompts for the selected metric (ARI, FKGL, etc.)
    metric_prompts = user_prompts.get(metric_name)

    if metric_prompts:
        prompt_index = {
            "no_token": 0,
            "token": 1,
            "token_explanation": 2,
            "token_explanation_examples": 3
        }.get(user_prompt_id, 0)  # Default to 0 if not found
        
        prompt_id = metric_prompts[prompt_index]["id"]
        prompt_text = metric_prompts[prompt_index]["prompt"]

        if "explanation" in user_prompt_id and explanation:
            prompt_text = prompt_text.replace("{EXPLANATION}", explanation)
        
        if "examples" in user_prompt_id and examples:
            for i, example in enumerate(examples, start=1):
                prompt_text = prompt_text.replace(f"{{EXAMPLE_{i}}}", example)

        prompt_text = prompt_text.replace("{SOURCE_VALUE}", str(source_metric_value))
        prompt_text = prompt_text.replace("{TARGET_VALUE}", str(target_metric_value))
        prompt_text = prompt_text.replace("{TEXT}", text)
        prompt_text = prompt_text.replace("{EOT_TOKEN}", "<|eot_id|>")
        # print(f"Prompt before return: {prompt_text}") # for debugging
        return prompt_id, prompt_text
    
    # Fallback: If the metric is not found, return a default prompt
    return user_prompt_id, f"Simplify the following text: \n\n{text}"

def format_prompt_with_special_tokens(tokenizer, system_prompt, user_prompt, metric_name, target_metric_value, model_family="base"):
    eos = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else ""
    bos = tokenizer.bos_token if hasattr(tokenizer, "bos_token") else ""
    control_token = f"<{metric_name}={target_metric_value}> "
    
    if model_family in ["llama", "mistral"]:
        formatted_prompt = (
            f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n"
            f"{user_prompt.strip()} [/INST] "
            f"{control_token}"
        )
    elif model_family == "qwen":
        formatted_prompt = (
            f"<|im_start|>system\n{system_prompt.strip()}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt.strip()}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{control_token}"
        )
    elif model_family == "base":
        formatted_prompt = (
            f"{bos}"
            f"<|start_header_id|>system<|end_header_id|> {system_prompt}{eos}\n"
            f"<|start_header_id|>user<|end_header_id|> {user_prompt}{eos}\n"
            f"<|start_header_id|>assistant<|end_header_id|> {control_token}"
        )
        
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    return formatted_prompt

def format_completion_with_special_tokens(tokenizer, completion, model_family="base"):
    eos = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else ""
    bos = tokenizer.bos_token if hasattr(tokenizer, "bos_token") else ""
    if model_family in ["llama", "mistral"]:
        return completion.strip()
    else:
        return f"{completion.strip()}{eos}"


def format_prompt_with_tokenizer(tokenizer, system_prompt, user_prompt, metric_name, target_metric_value):
    control_token = f"<{metric_name}={target_metric_value}>"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": control_token}
    ]

    # 制御トークンはモデルが書く簡約文の「接頭辞」でなければ条件付けとして効かない。
    # add_generation_prompt=True だと最後の assistant ターンが <|eot_id|> で閉じられ、
    # さらに assistant ヘッダが開き直されるため、制御トークンが簡約文と別ターンに分離する。
    # continue_final_message=True は最後の assistant ターンを閉じずに残す（両者は排他）。
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        continue_final_message=True
    )

    return formatted_prompt

def format_completion_with_tokenizer(tokenizer, completion):
    # プロンプトは制御トークン（例 <FKGL=9.6>）で終わる。chat template が assistant の
    # content を trim するため末尾の空白は残らない。ここで先頭に空白を補い
    # "<FKGL=9.6> The risk ..." と繋ぐ。空白の有無でトークンIDが変わる（"By"=1383 / " By"=3296）。
    return f" {completion.strip()}{tokenizer.eos_token}"
