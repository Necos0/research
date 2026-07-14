import os
import argparse
import torch
from tqdm import tqdm
import json
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
# from peft import PeftModel, PeftConfig
# from peft import LoraConfig, PeftModelForCausalLM, get_peft_config

from helpers.prompting import select_random_system_prompt, select_random_user_prompt, select_control_token_explanation, select_random_control_token_examples
from helpers.prompting import create_user_prompt, format_prompt_with_special_tokens, format_completion_with_special_tokens, format_prompt_with_tokenizer, format_completion_with_tokenizer
from helpers.hugging_face import load_dataset_from_hf, get_model_short_name
from classes.Metrics import Metrics

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def setup_tokenizer(model_source, model_family, max_length, fallback_source=None):
    tokenizer_source = model_source

    # Some saved checkpoints do not contain tokenizer files; fallback to base model tokenizer.
    if os.path.isdir(model_source):
        has_tokenizer_files = any(
            os.path.exists(os.path.join(model_source, name))
            for name in ["tokenizer.model", "tokenizer.json", "vocab.json"]
        )
        if not has_tokenizer_files and fallback_source:
            print(f"[WARN] Missing tokenizer files in {model_source}. Falling back to tokenizer from {fallback_source}.")
            tokenizer_source = fallback_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    # If max_length is -1, use the model's native max length
    if max_length == -1:
        # Keep the tokenizer's default max length from the model config
        print(f"Using model's native max length: {tokenizer.model_max_length}")
    else:
        tokenizer.model_max_length = max_length
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token is None:
        if model_family == "qwen":
            tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})
        elif model_family == "base":
            tokenizer.add_special_tokens({'eos_token': '<|eot_id|>'})

    return tokenizer

def load_and_prepare_model(model_name, model_family, model_path, model_class, max_length, peft_path=None):

    tokenizer = setup_tokenizer(model_path, model_family, max_length, fallback_source=model_name)

    if peft_path is not None:
        base_model_class = LlamaForCausalLM if model_class == "llama" else AutoModelForCausalLM
        base_model = base_model_class.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16
        )
        base_model.resize_token_embeddings(len(tokenizer))
        # from peft import PeftModel
        # model = PeftModel.from_pretrained(base_model, peft_path)
        from peft import PeftModelForCausalLM
        model = PeftModelForCausalLM.from_pretrained(base_model, peft_path)
        # model.resize_token_embeddings(len(tokenizer))

        if model_class == "llama":
            base_model = LlamaForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                # torch_dtype=torch.float16
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                # torch_dtype=torch.float16
            )

        # base_model.resize_token_embeddings(len(tokenizer))

    else:
        # Check if model_path has no model files but has checkpoint subdirectories
        # This happens when the training job only saved checkpoints, not final output
        actual_model_path = model_path
        if not os.path.exists(os.path.join(model_path, "model.safetensors.index.json")) and \
           not os.path.exists(os.path.join(model_path, "model.safetensors")) and \
           not os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json")):
            # Look for checkpoint-* directories
            checkpoint_dirs = [d for d in os.listdir(model_path) 
                             if os.path.isdir(os.path.join(model_path, d)) and d.startswith("checkpoint-")]
            if checkpoint_dirs:
                # Use the latest checkpoint (highest number)
                latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
                actual_model_path = os.path.join(model_path, latest_checkpoint)
                print(f"[INFO] No model weights at top level; auto-loading from checkpoint: {latest_checkpoint}")
        
        model_class = LlamaForCausalLM if model_class == "llama" else AutoModelForCausalLM
        model = model_class.from_pretrained(
            actual_model_path,
            device_map="auto",
            # torch_dtype=torch.float16
            )
        model.resize_token_embeddings(len(tokenizer))

    # 推論では KV キャッシュを必ず使う（False は学習時の設定。生成で無効だと
    # 毎ステップ全系列を再計算し、メモリと時間が系列長の二乗で膨らむ）
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def is_source_metric(args):
    if args.metric_name in ["FRE", "FKGL", "ARI", "DALE-CHALL", "Dale-Chall", "KEEP"]:
        return True
    return False
    
def is_compression_metric(args):
    if args.metric_name in ["CHAR_COMPRESSION", "WORD_COMPRESSION", "SENTENCE_COMPRESSION"]:
        return True
    return False

def load_and_prepare_test_set(dataset_name, tokenizer, max_length, control_tokens, system_prompts, user_prompts, metric_mapping, args, user_prompt_id, model_family, slice_test=None, source_based_metric=False, compression_metric=False):
    test_dataset = load_dataset_from_hf(dataset_name, split="test", slice=slice_test, local_data_dir=args.local_data_dir)
    
    def process_instance(row):

        system_id, system_prompt = select_random_system_prompt(system_prompts)

        metric_key_mapped = metric_mapping[args.metric_name]

        if source_based_metric:
            source_metric_value = row["source_metrics"][metric_key_mapped]
            target_metric_value = row["target_metrics"][metric_key_mapped]
        else:
            source_metric_value = None
            if args.metric_name == "WORD_COMPRESSION":
                target_metric_value = row["target_metrics"]["word_compression_rate"]
            elif args.metric_name == "CHAR_COMPRESSION":
                target_metric_value = row["target_metrics"]["char_compression_rate"]
            elif args.metric_name == "SENTENCE_COMPRESSION":
                target_metric_value = row["target_metrics"]["sentence_compression_rate"]

        reference_simplification = row["simplification_text"]
        
        # Dynamic explanation and examples if needed
        explanation = select_control_token_explanation(control_tokens, args.metric_name, target_metric_value) \
            if "explanation" in user_prompt_id else None
        
        examples = select_random_control_token_examples(control_tokens, args.metric_name) \
            if "examples" in user_prompt_id else None
        
        # Create the user prompt dynamically
        _, user_prompt = create_user_prompt(
            user_prompts=user_prompts,
            metric_name=args.metric_name,
            source_metric_value=source_metric_value,
            target_metric_value=target_metric_value,
            user_prompt_id=user_prompt_id,
            text=row["source_text"],
            explanation=explanation,
            examples=examples
        )
        
        # Format the prompt with special tokens
        # formatted_prompt = format_prompt_with_special_tokens(
        formatted_prompt = format_prompt_with_tokenizer(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metric_name=args.metric_name,
            target_metric_value=target_metric_value,
            # model_family=model_family
        )
        
        # Format the completion
        # formatted_completion = format_completion_with_special_tokens(
        formatted_completion = format_completion_with_tokenizer(
            tokenizer=tokenizer,
            completion=reference_simplification, 
            # model_family=model_family
            )

        # Handle tokenization with proper max_length (-1 means use model default)
        tokenizer_kwargs = {
            "truncation": True if max_length > 0 else False,
            "padding": "max_length" if max_length > 0 else False,
            "return_tensors": "pt",
            # prompt は apply_chat_template が既に BOS を含む。学習側（sft_finetune.py）と
            # 揃えて二重付与を防ぐ。
            "add_special_tokens": False
        }
        if max_length > 0:
            tokenizer_kwargs["max_length"] = max_length
        
        encoded = tokenizer(formatted_prompt, **tokenizer_kwargs)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        encoded_completion = tokenizer(formatted_completion, **tokenizer_kwargs)
        completion_ids = encoded_completion["input_ids"]

        return {
            "prompt": formatted_prompt,
            "completion": formatted_completion,
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "completion_ids": completion_ids.squeeze(0),  # Remove batch dimension,
            "attention_mask": attention_mask.squeeze(0),  # Remove batch dimension
        }

    # load_from_cache_file=False は必須。理由は sft_finetune.py の同じ箇所のコメントを参照
    # （prompting.py だけを直しても map のフィンガープリントが変わらず、古いキャッシュで
    #   プロンプトが組み立てられる。学習側がこれで壊れた）。
    test_dataset = test_dataset.map(process_instance, batched=False, load_from_cache_file=False)

    _sample = test_dataset[0]["prompt"]
    assert _sample.count("<|start_header_id|>assistant<|end_header_id|>") == 1, (
        "プロンプトに assistant ターンが2つある＝制御トークンが簡約文と別ターンに分離している。"
        f"\n{_sample[-200:]}"
    )
    assert _sample.rstrip().endswith(">") and f"<{args.metric_name}=" in _sample.rsplit("assistant<|end_header_id|>", 1)[-1], (
        "プロンプトが制御トークンで終わっていない＝簡約文の接頭辞として効かない。"
        f"\n{_sample[-200:]}"
    )
    print(f"--- prompt sanity OK: ...{_sample[-60:]!r}")

    return test_dataset

def run_inference(args, metric_mapping, model, tokenizer, test_dataset, batch_size=4, device="cuda", max_length=512, max_new_tokens=511, source_based_metric=False, compression_metric=False):
    model.eval()
    predictions = []    
    
    if args.temperature == 0.0:
        print(f"Using greedy decoding (temperature=0.0, do_sample=False)")
    elif args.temperature == -1:
        print(f"Using model's default temperature settings")
    else:
        print(f"Using temperature: {args.temperature}, do_sample: True")
    
    # Create a DataLoader to handle batching
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Running inference on test set"):
        batch = test_dataset[i:i + batch_size]
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

        # Pad sequences dynamically to the max length in the batch
        input_ids_list = [torch.tensor(item["input_ids"]) for item in batch]
        attention_mask_list = [torch.tensor(item["attention_mask"]) for item in batch] if "attention_mask" in batch[0] else None
        
        # Use torch.nn.utils.rnn.pad_sequence for dynamic padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        ).to(device)
        
        if attention_mask_list:
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_mask_list, 
                batch_first=True, 
                padding_value=0
            ).to(device)
        else:
            attention_mask = None
        
        with torch.no_grad():
            torch.cuda.empty_cache()

            # Generate with the max_new_tokens to limit the number of tokens generated beyond the input length
            # Temperature: 0.0 = greedy, -1 = model default, >0 = sampling with specified temp
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True
            }
            
            # Only set max_length if we have a specific limit (not using model default)
            if max_length > 0:
                gen_kwargs["max_length"] = max_length + max_new_tokens
            
            if args.temperature == 0.0:
                # Greedy decoding
                gen_kwargs["do_sample"] = False
            elif args.temperature == -1:
                # Use model defaults (don't set do_sample or temperature)
                pass
            else:
                # Use specified temperature
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
            
            outputs = model.generate(**gen_kwargs)
            
            # Slice off the generated portion (remove the input tokens)
            generated_only_ids = outputs[:, input_ids.shape[-1]:]  # Skip the input portion
            
            decoded_preds = tokenizer.batch_decode(
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up role tokens and extra whitespace from predictions
            cleaned_preds = []
            for pred in decoded_preds:
                # Remove common role tokens that models might generate
                pred_clean = pred.strip()
                # Remove "assistant" prefix (case-insensitive) followed by newlines/spaces
                if pred_clean.lower().startswith("assistant"):
                    pred_clean = pred_clean[len("assistant"):].lstrip()
                # Remove any leading special tokens or markers
                pred_clean = pred_clean.lstrip("<|").lstrip()
                cleaned_preds.append(pred_clean)
            
            print("\n--- Processing batch:")
            for item, pred in zip(batch, cleaned_preds):
                prediction_metrics = Metrics(input_text=pred.strip(), reference_text=item["simplification_text"], source_text=item["source_text"])
                computed_prediction_metrics = prediction_metrics.compute_metrics()
                source_metrics = Metrics(input_text=item["source_text"])
                computed_source_metrics = source_metrics.compute_metrics()
                reference_metrics = Metrics(input_text=item["simplification_text"], source_text=item["source_text"])
                computed_reference_metrics = reference_metrics.compute_metrics()

                prediction_char_compression = round(computed_prediction_metrics["char_count"] / computed_source_metrics["char_count"], 1)
                prediction_word_compression = round(computed_prediction_metrics["word_count"] / computed_source_metrics["word_count"], 1)
                prediction_sent_compression = round(computed_prediction_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
                computed_prediction_metrics["char_compression_rate"] = prediction_char_compression
                computed_prediction_metrics["word_compression_rate"] = prediction_word_compression
                computed_prediction_metrics["sentence_compression_rate"] = prediction_sent_compression

                reference_char_compression = round(computed_reference_metrics["char_count"] / computed_source_metrics["char_count"], 1)
                reference_word_compression = round(computed_reference_metrics["word_count"] / computed_source_metrics["word_count"], 1)
                reference_sent_compression = round(computed_reference_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
                computed_reference_metrics["char_compression_rate"] = reference_char_compression
                computed_reference_metrics["word_compression_rate"] = reference_word_compression
                computed_reference_metrics["sentence_compression_rate"] = reference_sent_compression

                prediction_dict = {
                    "global_id": item["global_id"],
                    "control_token": f"{args.metric_name}={computed_reference_metrics[metric_mapping[args.metric_name]]}",
                    "metric_name": args.metric_name,
                    "reference_metric_value": computed_reference_metrics[metric_mapping[args.metric_name]],
                    "source_text": item["source_text"],
                    "reference_simplification": item["simplification_text"],
                    "prediction": pred.strip(),
                    "prompt": item["prompt"],
                    "source_metrics": computed_source_metrics,
                    "prediction_metrics": computed_prediction_metrics,
                    "reference_metrics": computed_reference_metrics,
                    }
                if source_based_metric:
                    prediction_dict["source_metric_value"] = item["source_metrics"][metric_mapping[args.metric_name]]
                
                predictions.append(prediction_dict)

                print(f"{pred.strip()[:100]}...")

    return predictions

def run_inference_with_vllm(args, metric_mapping, llm, tokenizer, test_dataset, max_new_tokens, source_based_metric=False, compression_metric=False):
    # Build sampling params based on temperature setting
    sampling_kwargs = {
        "max_tokens": max_new_tokens,
        "stop": tokenizer.eos_token or None,
    }
    
    if args.temperature == 0.0:
        # Greedy decoding
        sampling_kwargs["temperature"] = 0.0
        sampling_kwargs["top_p"] = 1.0
        sampling_kwargs["top_k"] = -1
        print(f"Using greedy decoding (temperature=0.0)")
    elif args.temperature == -1:
        # Use vLLM defaults (don't specify temperature, top_p, top_k)
        print(f"Using vLLM's default sampling parameters")
    else:
        # Use specified temperature
        sampling_kwargs["temperature"] = args.temperature
        sampling_kwargs["top_p"] = 1.0
        sampling_kwargs["top_k"] = -1
        print(f"Using temperature: {args.temperature}")
    
    sampling_params = SamplingParams(**sampling_kwargs)

    prompts = [item["prompt"] for item in test_dataset]
    outputs = llm.generate(prompts, sampling_params)
    predictions = []

    print("\n--- Running vLLM generation")
    for item, output in zip(test_dataset, outputs):
        pred = output.outputs[0].text.strip()
        
        # Clean up role tokens and extra whitespace from predictions
        if pred.lower().startswith("assistant"):
            pred = pred[len("assistant"):].lstrip()
        # Remove any leading special tokens or markers
        pred = pred.lstrip("<|").lstrip()
        
        prediction_metrics = Metrics(input_text=pred, reference_text=item["simplification_text"], source_text=item["source_text"])
        computed_prediction_metrics = prediction_metrics.compute_metrics()
        source_metrics = Metrics(input_text=item["source_text"])
        computed_source_metrics = source_metrics.compute_metrics()
        reference_metrics = Metrics(input_text=item["simplification_text"], source_text=item["source_text"])
        computed_reference_metrics = reference_metrics.compute_metrics()

        # Compute compression
        prediction_char_compression = round(computed_prediction_metrics["char_count"] / computed_source_metrics["char_count"], 1)
        prediction_word_compression = round(computed_prediction_metrics["word_count"] / computed_source_metrics["word_count"], 1)
        prediction_sent_compression = round(computed_prediction_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
        computed_prediction_metrics["char_compression_rate"] = prediction_char_compression
        computed_prediction_metrics["word_compression_rate"] = prediction_word_compression
        computed_prediction_metrics["sentence_compression_rate"] = prediction_sent_compression

        reference_char_compression = round(computed_reference_metrics["char_count"] / computed_source_metrics["char_count"], 1)
        reference_word_compression = round(computed_reference_metrics["word_count"] / computed_source_metrics["word_count"], 1)
        reference_sent_compression = round(computed_reference_metrics["sent_count"] / computed_source_metrics["sent_count"], 1)
        computed_reference_metrics["char_compression_rate"] = reference_char_compression
        computed_reference_metrics["word_compression_rate"] = reference_word_compression
        computed_reference_metrics["sentence_compression_rate"] = reference_sent_compression

        prediction_dict = {
            "global_id": item["global_id"],
            "control_token": f"{args.metric_name}={computed_reference_metrics[metric_mapping[args.metric_name]]}",
            "metric_name": args.metric_name,
            "reference_metric_value": computed_reference_metrics[metric_mapping[args.metric_name]],
            "source_text": item["source_text"],
            "reference_simplification": item["simplification_text"],
            "prediction": pred,
            "prompt": item["prompt"],
            "source_metrics": computed_source_metrics,
            "prediction_metrics": computed_prediction_metrics,
            "reference_metrics": computed_reference_metrics,
        }
        if source_based_metric:
            prediction_dict["source_metric_value"] = item["source_metrics"][metric_mapping[args.metric_name]]

        predictions.append(prediction_dict)

        print(pred[:100] + "...")

    return predictions

def save_predictions_as_json(predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([p for p in predictions], f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for fast inference.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model dir.")
    parser.add_argument("--model_name", type=str, required=False, help="The name of the model on Hugging Face.")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset on Hugging Face.")
    parser.add_argument("--local_data_dir", type=str, default="data/splits_flattened_full",
                        help="ローカルスプリットのルート（フィルタ済みは data/splits_flattened_filtered）。存在しなければ HF Hub から取得。")
    parser.add_argument("--model_class", type=str, required=True, choices=["llama", "auto"], help="Model class to use.")
    parser.add_argument("--model_family", type=str, default="base", choices=["llama", "mistral", "qwen", "base"], help="Model family to use.")
    parser.add_argument("--max_length", type=int, default=512, help="Max length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--slice_test", type=int, default=-1, help="Slice the test set for dev. -1 means no slicing.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the predictions.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run inference on.")
    parser.add_argument("--peft_path", type=str, default=None, help="Path to PEFT adapter dir.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling. 0.0=greedy, -1=model default, >0=sampling with specified value.")

    
    # Arguments for dynamic prompting
    parser.add_argument("--control_tokens", type=str, required=True, help="Path to the control tokens JSON file.")
    parser.add_argument("--system_prompts", type=str, required=True, help="Path to the system prompts JSON file.")
    parser.add_argument("--user_prompts", type=str, required=True, help="Path to the user prompts JSON file.")
    parser.add_argument("--metric_mapping", type=str, required=True, help="Path to the metric mapping JSON file.")
    parser.add_argument("--metric_name", type=str, required=True, help="Metric name to use.")
    parser.add_argument("--user_prompt_id", type=str, default="token_explanation", choices=["no_token", "token", "token_explanation", "token_explanation_examples"], help="The user prompt ID to use.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Recover base model name from training metadata when not provided at inference time.
    if not args.model_name:
        args_json_path = os.path.join(args.model_path, "args.json")
        if os.path.exists(args_json_path):
            try:
                with open(args_json_path, "r", encoding="utf-8") as f:
                    saved_args = json.load(f)
                args.model_name = saved_args.get("model_name")
                if args.model_name:
                    print(f"[INFO] Loaded model_name from {args_json_path}: {args.model_name}")
            except Exception as e:
                print(f"[WARN] Failed to read {args_json_path}: {e}")

    print(f"Loading from {args.model_path}...")
    print(f"Inference args:\n{args}\n")
    set_seed(args.seed)


    if args.use_vllm:
        from vllm import LLM, SamplingParams
        print("Using vLLM for inference...")
        # Initialize vLLM
        llm = LLM(model=args.model_path, device=args.device)
        tokenizer = setup_tokenizer(args.model_path, args.model_family, args.max_length)
        model = None
    else:
        model, tokenizer = load_and_prepare_model(
            args.model_name,
            args.model_family, 
            args.model_path, 
            args.model_class, 
            args.max_length,
            peft_path=args.peft_path
            )
        model.to(args.device)

    # Load and prepare the dynamic prompting information (control tokens, system prompts, etc.)
    control_tokens = load_json(args.control_tokens)
    system_prompts = load_json(args.system_prompts)
    user_prompts = load_json(args.user_prompts)
    metric_mapping = load_json(args.metric_mapping)

    # check if source value exists
    source_based_metric = is_source_metric(args)
    # check if compression rate needs to be calculated
    compression_metric = is_compression_metric(args)

    # Load and prepare the test dataset with dynamic prompts
    test_dataset = load_and_prepare_test_set(
        args.dataset_name,
        tokenizer,
        args.max_length,
        control_tokens,
        system_prompts,
        user_prompts,
        metric_mapping,
        args,
        args.user_prompt_id,
        args.model_family,
        args.slice_test,
        source_based_metric=source_based_metric,
        compression_metric=compression_metric
    )

    if args.use_vllm:
        predictions = run_inference_with_vllm(
            args,
            metric_mapping,
            llm,
            tokenizer,
            test_dataset,
            max_new_tokens=args.max_length - 1,
            source_based_metric=source_based_metric,
            compression_metric=compression_metric
        )
    else:
        predictions = run_inference(
            args, 
            metric_mapping, 
            model, 
            tokenizer, 
            test_dataset, 
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length, 
            max_new_tokens=args.max_length - 1,
            source_based_metric=source_based_metric,
            compression_metric=compression_metric
            )

    # Save the predictions to a file
    save_predictions_as_json(predictions, args.output_file)
    print(f"Predictions saved to {args.output_file}")

 
if __name__ == "__main__":
    main()
