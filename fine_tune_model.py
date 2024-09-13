import os
import json
import torch
from datasets import Dataset, DatasetDict
from transformers import BartTokenizerFast, BartForConditionalGeneration, TrainingArguments, Trainer

# Initialise tokeniser
tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large-cnn')    # BART's large CNN variant for text summarisation

# Load pre-trained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Load and tokenise data from JSON files
def load_and_tokenize_data(json_path, is_test=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    summaries = []

    # Extract input texts and summaries
    for item in data:
        text = item['input_text']
        tokenized_text = tokenizer(text, max_length=1024, padding='max_length', truncation=True)
        texts.append(tokenized_text)

        if not is_test:
            summary = item.get('target_summary')
            if summary is not None:
                tokenized_summary = tokenizer(summary, max_length=512, padding='max_length', truncation=True)
                summaries.append(tokenized_summary)
            else:
                raise ValueError("Missing 'target_summary' in the dataset for a non-test scenario.")
        else:
            # Test dataset does not have summaries
            summaries.append(None)

    return texts, summaries

# Load and tokenise datasets from JSON files
train_texts, train_summaries = load_and_tokenize_data('./data/train/train.json')
val_texts, val_summaries = load_and_tokenize_data('./data/val/val.json')
test_texts, _ = load_and_tokenize_data('./data/test/test.json', is_test=True)

# Convert tokenised data into Hugging Face's Dataset objects for easier processing
train_dataset = Dataset.from_dict({
    "input_ids": [t['input_ids'] for t in train_texts],
    "labels": [s['input_ids'] for s in train_summaries]
})
val_dataset = Dataset.from_dict({
    "input_ids": [t['input_ids'] for t in val_texts],
    "labels": [s['input_ids'] for s in val_summaries]
})
test_dataset = Dataset.from_dict({
    "input_ids": [t['input_ids'] for t in test_texts]
})

# Sanity check ensures datasets are correctly loaded and tokenised
assert len(train_dataset) > 0, "Train dataset is empty!"
assert len(val_dataset) > 0, "Validation dataset is empty!"
assert len(test_dataset) > 0, "Test dataset is empty!"

# Organise datasets into a single DatasetDict for convenient access during training
datasets = DatasetDict({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
})

# Loss function with anti-copy mechanism
# Helps model learn how to generate more original summaries by penalising excessive copying from input texts
def loss_function(outputs, labels, input_ids, alpha=0.5):
    ce_loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

    # Anti-copy penalty calculation
    generated_tokens = outputs.logits.argmax(dim=-1)
    generated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    penalty = compute_copy_penalty(input_text, generated_text)

    loss = ce_loss + alpha * penalty
    return loss

def compute_copy_penalty(input_text, generated_text):
    penalty = 0
    for in_text, gen_text in zip(input_text, generated_text):
        in_words = set(in_text.split())
        gen_words = set(gen_text.split())
        common_words = in_words.intersection(gen_words)
        penalty += len(common_words) / len(gen_words)
    return penalty / len(generated_text)

# Allows for integration of loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = loss_function(outputs, inputs['labels'], inputs['input_ids'])
        return (loss, outputs) if return_outputs else loss

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
)

# Initialise class with model, training arguments, and tokenised datasets
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    tokenizer=tokenizer
)

print("Starting training...")
trainer.train()

print("Starting evaluation...")
eval_results = trainer.evaluate(eval_dataset=datasets["val"])
print(f"Evaluation results: {eval_results}")

# Fine-tuned model and tokeniser saved to saved_model directory
save_directory = "./saved_model"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
