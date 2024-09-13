from transformers import BartTokenizer, BartForConditionalGeneration

# Directory where the model is saved
save_directory = "./saved_model"

# Load fine-tuned model and tokeniser
tokenizer = BartTokenizer.from_pretrained(save_directory)
model = BartForConditionalGeneration.from_pretrained(save_directory)
