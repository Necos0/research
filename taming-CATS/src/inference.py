from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "shtosti/Llama-3.2-1B-Instruct-med-fkgl"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", use_auth_token=True)

# Your prompt
prompt = "Explain the function of the kidney in simple terms."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
