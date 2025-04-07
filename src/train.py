import os
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from helpers import load_travel_data, get_tokenizer, tokenize_function

# Disable W&B
os.environ["WANDB_MODE"] = "disabled"

def main():
    model_name = "gpt2"
    tokenizer = get_tokenizer(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load and tokenize data
    dataset = load_travel_data("data/travel_data.json")
    tokenized_dataset = dataset.map(tokenize_function(tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        evaluation_strategy="no",
        logging_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        num_train_epochs=15,
        weight_decay=0.01,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # Set False if running on CPU
        report_to=None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    trainer.train()
    # Save the fine-tuned model
    model.save_pretrained("./model/fine_tuned_travel_gpt2")
    tokenizer.save_pretrained("./model/fine_tuned_travel_gpt2")

print("Fine-tuning completed and model saved!")

if __name__ == "__main__":
    main()