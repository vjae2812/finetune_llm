import json
from datasets import Dataset
from transformers import GPT2Tokenizer

def load_travel_data(file_path: str = "data/travel_data.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict({"text": [item["text"] for item in data]})

def get_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token
    return tokenizer

def tokenize_function(tokenizer):
    def tokenize(examples):
        encodings = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)
        encodings['labels'] = encodings['input_ids']
        return encodings
    return tokenize
