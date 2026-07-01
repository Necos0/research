from datasets import load_dataset
import os
from dotenv import load_dotenv
from datasets import Features, Value

load_dotenv(dotenv_path="./../.env", override=True)

hf_token = os.getenv("HF_TOKEN")
os.system(f"huggingface-cli login --token {hf_token}")

dataset = load_dataset(
    "shtosti/Med-EASi",
    data_files={"train": "train.jsonl", "test": "test.jsonl", "validation": "val.jsonl"}
)

validation_dataset = dataset["validation"]

print(len(validation_dataset))
for line in validation_dataset:
    print(line)
