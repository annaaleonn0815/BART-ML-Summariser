import os
import json
from transformers import BartTokenizerFast, BartForConditionalGeneration


model_dir = "./saved_model"

# Load fine-tuned model and tokeniser
tokenizer = BartTokenizerFast.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir)

# Summarise text using the fine-tuned model
def summarize_text(text):
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=512, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load input data from JSON file and generate summaries
def summarize_from_json(json_path, output_json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summaries = []
    for item in data:
        input_text = item['input_text']
        summary = summarize_text(input_text)
        summaries.append({
            "input_text": input_text,
            "generated_summary": summary
        })

    # Save summaries to an output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=4)

    print(f"Summarization completed. Results saved to {output_json_path}")

if __name__ == "__main__":
    input_json_path = "./data/test/test.json"  # Path to input JSON file
    output_json_path = "./data/test/test_summary.json"  # Save summarised output

    summarize_from_json(input_json_path, output_json_path)
