
import os
import json
import random
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="./../.env", override=True)
from classes.Metrics import Metrics

# Set seed for reproducibility
random.seed(42)

# Load OpenAI API Key and Model Config
OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
PROJECT = os.getenv("PROJECT_NAME")
MODEL = "gpt-4o-mini"
client = OpenAI(api_key=OPENAI_TOKEN)

def load_json(file_path: str):
    """Load JSON from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def load_dataset(dataset_name: str, split="dev") -> list:
    """Load dataset from a JSONL file (line-delimited JSON)."""
    file_path = f"./../data/splits_w_dev/{dataset_name}/{split}.jsonl"
    
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]

def generate_response(messages, model=MODEL):
    """Send a request to OpenAI's API."""
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def simplify(sys_prompt, user_prompt):
    """Simple text simplification process: immediate generation."""
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return generate_response(messages)

def simplify_with_reasoning(sys_prompt, user_prompt):
    """

    Two-turn text simplification process.

    includes 2-step process: 
        1. generate reasoning 
        2. generate simplification
    
    """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Here is a text simplification task: \n{user_prompt}. \n\nDo not yet generate the simplification. Just briefly think about how you would approach this simplification task. Think step by step, and perform the simplification calculations, if required. Be very brief."}
    ]
    reasoning = generate_response(messages)
    messages.append({"role": "assistant", "content": reasoning})
    messages.append({"role": "user", "content": f"Now that you have developed a simplification strategy, generate the simplification. Write only the simplification, no other comments are allowed."})

    return reasoning, generate_response(messages)

def simplify_with_transformations(sys_prompt, user_prompt):
    """
        two-turn text simplification process.

        Includes separate transformations: 

        - syntactic simplification
        - lexical simplification
        - paraphrase/explanation, 
        - final simplification.

        """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Here is a text simplification task: {user_prompt}."},
        {"role": "user", "content": f"Think about the simplification as a stepwise process. Think briefly about each step. \n\nSYNTACTIC: \nStart by performing a syntactic simplification. Reduce sentences to minimal clauses. You can split or merge sentences, if needed. \n\nLEXICAL: \nProceed with a lexical simplification. Substitute domain-specific and difficult words with simple words, if possible. \n\nPARAPHRASE: \Finally, feel free to use paraphrase, add explanations for terms and concepts you deem too difficult, or drop them altogether."},
    ]

    all_operations_combined = generate_response(messages)

    messages.append({"role": "assistant", "content": all_operations_combined})
    messages.append({"role": "user", "content": f"Generate the final simplification based on your previous ideas and user instructions. Output only the simplification. No notes or comments are allowed."})

    simplification = generate_response(messages)

    return all_operations_combined, simplification

def select_random_system_prompt(system_prompts):
    """Randomly selects a system prompt from the list and returns its ID and text."""
    selected_prompt = random.choice(system_prompts["system_prompts"])  # Seed set for reproducibility
    return selected_prompt["id"], selected_prompt["prompt"]

def select_random_user_prompt(user_prompts, metric_name, metric_value, user_prompt_id):
    """Randomly selects a user prompt from the list and returns its ID and text."""
    selected_prompts = user_prompts.get(metric_name)
    if selected_prompts:
        selected_prompt = random.choice(selected_prompts[user_prompt_id])
        return selected_prompt["id"], selected_prompt["prompt"].replace("{VALUE}", metric_value)
    return None, None

def select_control_token_explanation(control_tokens, metric_name, metric_value):
    """Selects a control token explanation based on the metric name and value."""
    control_token = control_tokens.get(metric_name)
    if control_token:
        return control_token["explanation"].replace("{VALUE}", metric_value)
    return None

def select_random_control_token_examples(control_tokens, metric_name, metric_value, num_examples=3):
    """Selects a random set of control token examples based on the metric name and value."""
    control_token = control_tokens.get(metric_name)
    if control_token and "examples" in control_token:
        # examples = random.sample(control_token["examples"], num_examples)
        examples = random.sample(control_token["examples"], min(num_examples, len(control_token["examples"])))
        return examples
    return []

def create_user_prompt(user_prompts, metric_name, metric_value, user_prompt_id, text, explanation=None, examples=None):
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
        
        # Replace the {VALUE} placeholder with the actual control token value
        return prompt_id, prompt_text.replace("{VALUE}", metric_value).replace("{TEXT}", text)
    
    # Fallback: If the metric is not found, return a default prompt
    return user_prompt_id, f"Simplify the following text: \n\n{text}"

def main():
    # TODO create mapping between metric names in the dataset and the ones used in the user prompts
    # TODO select user_prompt_id at random?

    # Define user prompt settings
    user_prompt_id = "token" # TODO select "no_token", "token", "token_explanation", "token_explanation_examples"
    metric_name = "ARI" # TODO change

    # Define experiment settings
    experiment_name = f"dynamic_prompting_{user_prompt_id}_{metric_name}" # TODO change
    dataset_name = "medeasi"

    # Select prompting type
    prompting_type = "transformations"  # TODO choose from "vanilla", "reasoning", "transformations"

    if prompting_type == "vanilla":
        simplify_fn = simplify
    elif prompting_type == "reasoning":
        simplify_fn = simplify_with_reasoning
    elif prompting_type == "transformations":
        simplify_fn = simplify_with_transformations
    

    # Load dataset
    data = load_dataset(dataset_name)
    data = data[10:13]  # For testing purposes TODO remove slice when done with testing

    # Load prompts and control tokens from JSON
    control_tokens = load_json("./../data/prompts/control_tokens.json")
    system_prompts = load_json("./../data/prompts/system_prompts.json")
    user_prompts = load_json("./../data/prompts/user_prompts.json")

    # Define output file path
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./../experiments/prompting/{experiment_name}/{prompting_type}/{MODEL}/{dataset_name}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/results.jsonl"

    with open(output_file, "w", encoding="utf-8") as jsonl_file:

        for item in data:
            input_text = item["source_text"]

            for simplification in item["simplifications"]:
                reference_simplified_text = simplification["simplification_text"]
                reference_metrics = simplification["target_metrics"]

                metric_value = reference_metrics.get(metric_name, None)

                if metric_value is None:
                    print(f"Metric {metric_name} not found for simplification {item['global_id']}")
                    continue
                
                # Select a control token explanation
                control_token_explanation = None
                if "explanation" in user_prompt_id:
                    control_token_explanation = select_control_token_explanation(control_tokens, metric_name, str(metric_value))
                    print(f"Control Token Explanation: {control_token_explanation}")

                control_token_examples = None
                if "examples" in user_prompt_id:
                    control_token_examples = select_random_control_token_examples(control_tokens, metric_name, str(metric_value))
                    print(f"Control Token Examples: {control_token_examples}")

                # Randomly select a system prompt
                system_prompt_id, system_prompt_text = select_random_system_prompt(system_prompts)
                # Create a user prompt
                user_prompt_id, user_prompt = create_user_prompt(
                                        user_prompts, 
                                        metric_name, 
                                        str(metric_value), 
                                        user_prompt_id, 
                                        input_text,
                                        explanation=control_token_explanation,
                                        examples=control_token_examples
                                        )

                # Generate simplified text
                if prompting_type == "vanilla":
                    generated_simplified_text = simplify_fn(
                                        sys_prompt=system_prompt_text, # selected randomly for robustness and variability
                                        user_prompt=user_prompt # includes controls and input text
                                        )
                elif prompting_type == "reasoning":
                    reasoning, generated_simplified_text = simplify_fn(
                                            sys_prompt=system_prompt_text,
                                            user_prompt=user_prompt
                                            )
                elif prompting_type == "transformations":
                    transformations, generated_simplified_text = simplify_fn(
                                            sys_prompt=system_prompt_text,
                                            user_prompt=user_prompt
                                            )

                # Compute metrics for the generated text
                generated_metrics = Metrics(
                                        input_text=generated_simplified_text,
                                        reference_text=reference_simplified_text,
                                        source_text=input_text
                                        ).compute_metrics()


                # Compute and store metrics for the original and simplified text
                output_data = {
                    "global_id": item["global_id"], # keep track of the original text
                    "experiment": experiment_name,
                    "model": MODEL,
                    "dataset": dataset_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "control_token": f"<{metric_name}={metric_value}>",
                    "prompting_type": prompting_type,
                    "sys_prompt_id": system_prompt_id,
                    "sys_prompt": system_prompt_text,
                    "user_prompt_id": user_prompt_id,
                    "user_prompt": user_prompt,
                    "reasoning": reasoning if prompting_type == "reasoning" else None,
                    "transformations": transformations if prompting_type == "transformations" else None,
                    "source_text": input_text,
                    "source_metrics": item["source_metrics"],
                    "target_text": generated_simplified_text,
                    "target_metrics": generated_metrics,
                    "reference_text": reference_simplified_text,
                    "reference_metrics": reference_metrics,
                }

                # Write the output data to a JSONL file
                jsonl_file.write(json.dumps(output_data) + "\n")

                print(f"[{item['global_id']}] Simplified: {generated_simplified_text[:100]}...")
                print(f"Saved output for {item['global_id']}")
                print()


if __name__ == "__main__":
    main()
