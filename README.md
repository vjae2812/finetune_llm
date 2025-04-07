# finetune_llm

# 🤖 GPT-2 Fine-tuning on Travel Data

This project demonstrates how to fine-tune OpenAI's GPT-2 model using Hugging Face Transformers on a small travel-related dataset. Useful for learning how to prepare data, tokenize, and train LLMs for domain-specific generation tasks.


--------------------------------------------------
Project Structure
--------------------------------------------------

finetune_llm
├── data/
│   └── travel_data.json              # Travel-related text data
├── models/                           # Saved model checkpoints
├── notebook                          # notebook experiment
├── scr/
   └── helpers.py                     # Data loading and tokenization utilities
   ├── train.py                       # Main training script
   ├── infer.py                       # Do to the inference
├── .gitignore
├── requirements.txt
└── README.txt                        # You're reading this!

--------------------------------------------------
Step-by-Step Instructions
--------------------------------------------------

1. Clone the Repository
-----------------------
git clone https://github.com/vjae2812/finetune_llm.git
cd finetune_llm

2. Set Up the Environment
-------------------------
(Optional) Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

3. Understand the Dataset
-------------------------
Open the file `data/travel_data.json`. It contains sample travel sentences like:

{"text": "Paris is a beautiful city known for its iconic Eiffel Tower."}

You can add more data in the same format.

4. Train the Model
------------------
Run the training script:

python train.py

What this script does:
- Loads the travel data.
- Tokenizes it using Hugging Face tokenizer.
- Fine-tunes GPT-2 using the Trainer API.
- Saves the model checkpoints in `models/fine_tuned_travel_gpt2`.

Note:
W&B logging is disabled (WANDB_MODE=disabled, report_to=None).

5. Model Output
---------------
After training completes, the model checkpoints will be saved to:

models/fine_tuned_travel_gpt2/

To load this model later:

from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("models/fine_tuned_travel_gpt2")

--------------------------------------------------
Additional Notes
--------------------------------------------------

- The tokenizer uses EOS token as padding since GPT-2 doesn't support pad_token.
- fp16 is set to False by default (for CPU). Set to True for GPU training.
- Hugging Face `datasets.Dataset` is used to keep everything in-memory.

--------------------------------------------------
To-Do / Future Ideas
--------------------------------------------------

- Add evaluation or test data
- Use larger or real-world travel datasets

--------------------------------------------------
Tech Stack
--------------------------------------------------

- Python
- Hugging Face Transformers
- Datasets
- PyTorch

--------------------------------------------------
License
--------------------------------------------------

MIT License – do whatever you want 🤘

--------------------------------------------------
Credits
--------------------------------------------------

Thanks to Hugging Face for providing open access to GPT models and training tools.

