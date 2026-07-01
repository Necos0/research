from transformers import TrainerCallback
import wandb
import torch
import random
import warnings

class PredictionLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset, log_every=20, num_samples=4, max_length=512, gen_kwargs=None):
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset
        self.log_every = log_every
        self.num_samples = num_samples
        self.max_length = max_length

        # Set pad_token_id if not already set (important for decoder-only models)
        if self.tokenizer.pad_token_id is None:
            warnings.warn("Tokenizer has no pad_token_id; setting pad_token_id to eos_token_id.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.padding_side = 'left'

        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 1500,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every != 0 or state.global_step == 0:
            return

        model = kwargs['model']
        model.eval()

        sample_indices = random.sample(range(len(self.val_dataset)), self.num_samples)

        prompts, references = [], []
        for idx in sample_indices:
            sample = self.val_dataset[idx]

            # Handle dict or tuple-style samples
            if isinstance(sample, dict):
                prompt = sample.get("prompt", "")
                reference = sample.get("completion", "")
            elif isinstance(sample, tuple):
                prompt = sample[0] if isinstance(sample[0], str) else str(sample[0])
                reference = sample[1] if len(sample) > 1 else ""
            else:
                continue

            if prompt:
                prompts.append(prompt)
                references.append(reference)

        if not prompts:
            return

        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **self.gen_kwargs
                )

            # Slice generated tokens to remove the prompt portion
            prompt_length = inputs["input_ids"].shape[1]
            generated_only_ids = output_ids[:, prompt_length:]

            decoded_preds = self.tokenizer.batch_decode(
                # output_ids,
                generated_only_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            predictions = []
            for step, prompt, reference, prediction in zip(
                [state.global_step] * len(prompts), prompts, references, decoded_preds
            ):
                print(f"\n>>> [Step {step}] Prompt:\n{prompt}")
                print(f">>> Reference:\n{reference}")
                print(f">>> Prediction:\n{prediction}")
                print("=" * 80)

                predictions.append([step, prompt, reference, prediction])

            # Log to wandb
            if predictions:
                table = wandb.Table(data=predictions, columns=["step", "prompt", "reference", "prediction"])
                wandb.log({"predictions": table}, step=state.global_step)

        except Exception as e:
            print(f"[ERROR] Batched generation failed at step {state.global_step}: {e}")
