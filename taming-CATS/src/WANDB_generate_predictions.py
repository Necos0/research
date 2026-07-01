import os
import argparse
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from helpers.hugging_face import load_dataset_from_hf
from helpers.prompting import create_inference_prompt
from datetime import datetime
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer directory or model name")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--metric_name", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None, help="Where to save predictions as JSON")
    parser.add_argument("--peft", action="store_true", help="Whether to load PEFT adapter")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_tokenizer(model_path, tokenizer_path, peft_enabled):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    if peft_enabled:
        print("Loading PEFT adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_predictions(model, tokenizer, dataset, metric_name, args):
    predictions = []

    for example in tqdm(dataset, desc="Generating predictions"):
        source_text = example["source_text"]
        metric_value = example["target_metrics"][metric_name]
        input_prompt = create_inference_prompt(source_text, metric_name, metric_value)

        input_ids = tokenizer(input_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
        
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        predictions.append({
            "input_prompt": input_prompt,
            "generated_output": decoded,
            "reference": example.get("simplification_text", None),
            "metric_value": metric_value
        })

    return predictions


def main():
    args = parse_args()
    load_dotenv()

    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path, args.peft)
    dataset = load_dataset_from_hf(args.dataset_name, split=args.split)

    predictions = generate_predictions(model, tokenizer, dataset, args.metric_name, args)

    output_file = args.output_file or f"./predictions_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    main()
