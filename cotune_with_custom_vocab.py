# Data loading
import pandas as pd
from datasets import Dataset, load_from_disk

# Tokenizer Definition
import json
from transformers import PreTrainedTokenizer

# Model training / generation
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from tqdm import tqdm


dataset_file_path = "encoded_dataset"
dataset = load_from_disk(dataset_file_path)


# Step 1: Load the JSON file containing the unique token vocabulary
with open('vocab_map.json', 'r') as f:
    token_vocab = json.load(f)
    
    
import re

# Step 1: Define special tokens
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'

# Step 2: Update your token vocabulary to include special tokens (if not already present)
token_vocab.extend([BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN])

# Step 3: Define a regular expression for detecting userIDs.
number_regex = re.compile(r'\d+')

# Step 4: Reinitialize the custom tokenizer to handle digit splitting
class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab, **kwargs):
        self.vocab = {token: i for i, token in enumerate(vocab)}
        super().__init__(**kwargs)
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

    def _tokenize(self, text):
        tokens = []
        # Split text into words
        words = text.split()

        for word in words:
            # If the word is a number, split into individual digits
            if number_regex.fullmatch(word):
                tokens.extend(list(word))  # Split the number into digits
            else:
                tokens.append(word)

        # Add BOS and EOS tokens
        tokens = [self.bos_token] + tokens + [self.eos_token]
        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

# Step 5: Initialize the tokenizer
tokenizer = CustomTokenizer(vocab=token_vocab)

# Step 6: Test tokenization with BOS, EOS, and number handling
text = dataset['test'][0]['text']
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens with BOS/EOS and number handling: {tokens}")
print(f"Token IDs: {token_ids}")


# Step 1: Tokenize the dataset and pad/truncate to a given max sequence length
def tokenize_function(examples, max_length):
    # Tokenize the text, ensure padding and truncation to max_length, including BOS/EOS tokens
    tokenized = tokenizer(
        examples["text"],
        truncation=True,        # Truncate sequences longer than max_length
        padding="max_length",   # Pad sequences shorter than max_length
        max_length=max_length,  # Define the max length
        add_special_tokens=True # Add BOS/EOS tokens
    )
    
    # In autoregressive training, the labels are the same as the input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# Step 2: Define your max_length 
# Max columns in dataset = 120
max_length = 128 

# Step 3: Apply the tokenizer to the dataset, ensuring all examples are padded to max_length
tokenized_datasets = dataset.map(lambda x: tokenize_function(x, max_length), batched=True)


# Step 2: Define the GPT-2 model architecture for a distill model
# You can configure the distillation process by reducing the number of layers, heads, etc.
config = GPT2Config(
    vocab_size=len(tokenizer.get_vocab()),
    n_embd=256,  # Smaller embedding size for distillation
    n_layer=6,   # Fewer layers than standard GPT-2
    n_head=4,    # Fewer attention heads
    n_positions=max_length,  # Position embeddings
)

# Initialize a new GPT-2 model with the custom configuration
model = GPT2LMHeadModel(config)

# Step 3: Set up training arguments
training_args = TrainingArguments(
    output_dir="checkpoints/gpt2-distilled",
    overwrite_output_dir=True,
    num_train_epochs=350,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=10,
    logging_dir="./logs",
    logging_steps=500,
    evaluation_strategy="steps",
    eval_steps=1000,
    load_best_model_at_end=True
)

# Step 4: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Optional, if validation set is available
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the model
trainer.save_model("checkpoints/gpt2-distilled")


# Step 1: Function to generate sentences based on first half of input_ids
def complete_sentences(model, tokenizer, tokenized_dataset, max_length):
    completed_sentences = []
    
    # Ensure model is in evaluation mode
    model.eval()
    
    for example in tqdm(tokenized_dataset):
        input_ids = example['input_ids']
        
        # Step 2: Take the first half of the input_ids as the prompt
        half_length = len(input_ids) // 2
        prompt_ids = input_ids[:half_length]
        
        # Step 3: Use the model to generate the complete sentence
        input_ids_tensor = torch.tensor([prompt_ids]).to(model.device)  # Add batch dimension
        generated_ids = model.generate(
            input_ids=input_ids_tensor,
            max_length=max_length,  # Generate up to the max length
            pad_token_id=tokenizer.pad_token_id,  # Ensure proper padding handling
            eos_token_id=tokenizer.eos_token_id  # Stop at EOS token
        )
        
        # Step 4: Convert generated token IDs back to text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        completed_sentences.append(generated_text)
    
    return completed_sentences

# Generate completed sentences as text
completed_sentences = complete_sentences(model, tokenizer, tokenized_datasets['test'], max_length)

# Completed sentences will now be a list of text where each element is a completed sentence
#print(completed_sentences)



import re

def combine_digits(sentence):
    # Use regex to find sequences of single digits and combine them into a single number
    processed_sentence = re.sub(r'(?<=\b)(\d\s)+\d(?=\b)', lambda x: ''.join(x.group(0).split()), sentence)
    return processed_sentence

def process_completed_sentences(completed_sentences):
    # Apply the combine_digits function to each sentence
    processed_sentences = [combine_digits(sentence) for sentence in completed_sentences]
    return processed_sentences

processed_sentences = process_completed_sentences(completed_sentences)
print(processed_sentences[0])


import os
# Function to save completed sentences to a file
def save_sentences_to_file(completed_sentences, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in completed_sentences:
            f.write(sentence + '\n')  # Write each sentence on a new line

syn_dir = "synth_data"
os.makedirs(syn_dir, exist_ok=True)
file_path = 'conditional_generation.txt'  # Specify the file path
save_sentences_to_file(processed_sentences, os.path.join(syn_dir,file_path))

print(f"Completed sentences have been saved to {file_path}")

