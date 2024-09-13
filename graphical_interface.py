import os
import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import BartTokenizerFast, BartForConditionalGeneration

# Load fine-tuned model and tokenizer
def summarize_text(text, summary_length):
    model_dir = './saved_model'
    tokenizer = BartTokenizerFast.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)

    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

    # Generate summary
    summary_ids = model.generate(
        inputs.input_ids,
        do_sample=False,
        num_beams=5,
        max_length=summary_length,
        min_length=max(100, int(0.8 * summary_length)),
        length_penalty=0.4,
        no_repeat_ngram_size=3,
        early_stopping=False
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to load input text from file
def load_file():
    file_path = filedialog.askopenfilename(initialdir="cli_tool_documents", title="Select a file",
                                           filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    if file_path:
        with open(file_path, "r") as file:
            input_text.delete(1.0, tk.END)
            input_text.insert(tk.END, file.read())
        generate_button.config(state="normal")  # Enable generate button after loading text

# Function to generate and display summary
def generate_summary():
    text = input_text.get(1.0, tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please load a text file or enter text to summarize.")
        return

    try:
        summary_length = int(summary_length_var.get())
    except ValueError:
        summary_length = 150

    summary = summarize_text(text, summary_length)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, summary)
    save_button.config(state="normal")  # Enable save button after generating summary

# Function to save generated summary to a file
def save_summary():
    summary = output_text.get(1.0, tk.END).strip()
    if not summary:
        messagebox.showwarning("Save Error", "No summary available to save.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, "w") as file:
            file.write(summary)
        messagebox.showinfo("Success", f"Summary saved to {file_path}")

# GUI Setup with styling
root = tk.Tk()
root.title("Text Summarization Tool")
root.geometry("700x600")
root.configure(bg="#f0f0f0")

# Input Text Section
input_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
input_frame.pack(fill=tk.BOTH)

input_label = tk.Label(input_frame, text="Input Text", font=("Helvetica", 12), bg="#f0f0f0")
input_label.pack(anchor=tk.W, padx=10)

input_text = tk.Text(input_frame, height=10, width=80, wrap=tk.WORD, font=("Helvetica", 10))
input_text.pack(padx=10)

load_button = tk.Button(input_frame, text="Load Text File", command=load_file, font=("Helvetica", 10), bg="#007BFF", fg="black")
load_button.pack(pady=5)

# Summary Length Section
summary_length_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
summary_length_frame.pack(fill=tk.BOTH)

summary_length_label = tk.Label(summary_length_frame, text="Summary Length", font=("Helvetica", 12), bg="#f0f0f0")
summary_length_label.pack(anchor=tk.W, padx=10)

summary_length_var = tk.StringVar(value="150")
summary_length_entry = tk.Entry(summary_length_frame, textvariable=summary_length_var, font=("Helvetica", 10), width=10)
summary_length_entry.pack(padx=10)

# Generate Button
generate_button = tk.Button(summary_length_frame, text="Generate Summary", command=generate_summary, font=("Helvetica", 10), bg="#28A745", fg="black", state="disabled")
generate_button.pack(pady=5)

# Output Text Section
output_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
output_frame.pack(fill=tk.BOTH)

output_label = tk.Label(output_frame, text="Generated Summary", font=("Helvetica", 12), bg="#f0f0f0")
output_label.pack(anchor=tk.W, padx=10)

output_text = tk.Text(output_frame, height=10, width=80, wrap=tk.WORD, font=("Helvetica", 10))
output_text.pack(padx=10)

# Save Button
save_button = tk.Button(output_frame, text="Save Summary", command=save_summary, font=("Helvetica", 10), bg="#FFC107", fg="black", state="disabled")
save_button.pack(pady=5)

# Start GUI
root.mainloop()
