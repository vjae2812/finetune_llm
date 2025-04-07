from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./model/fine_tuned_travel_gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("./model/fine_tuned_travel_gpt2")

# Set the model to evaluation mode
model.eval()

# Example prompt (You can change this to any travel-related prompt)
prompt = "The best places to visit in Europe are"

# Encode the prompt text
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    input_ids,
    max_length=100,       # maximum length of the generated text
    num_beams=5,          # number of beams for beam search (higher = more diverse)
    no_repeat_ngram_size=2,  # Prevent repeating n-grams
    temperature=0.7,      # Controls randomness in predictions (lower = more focused)
    top_k=50,             # Limits the number of highest probability tokens to consider
    top_p=0.95,           # Uses nucleus sampling (top-p sampling)
    do_sample=True,       # Whether to use sampling or greedy decoding
    early_stopping=True   # Stop early when an end token is reached
)

# Decode the generated output back to text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Prompt:", prompt)
print("Generated Text:", generated_text)
