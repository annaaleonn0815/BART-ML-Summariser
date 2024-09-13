import json
from rouge_score import rouge_scorer
import numpy as np

def calculate_rouge_scores(pretrained_path, reference_path, output_file):
    # Load generated summaries and reference summaries
    with open(pretrained_path, 'r', encoding='utf-8') as f:
        pretrained_summaries = json.load(f)

    with open(reference_path, 'r', encoding='utf-8') as f:
        reference_summaries = json.load(f)

    # Initialise ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Lists to store scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Calculate ROUGE scores for each pair of summaries
    for pretrained, reference in zip(pretrained_summaries, reference_summaries):
        # Extract summaries using correct keys
        gen_summary = pretrained.get('generated_summary', '')
        ref_summary = reference.get('reference_summary', '')

        # Check if either summary is missing or empty
        if not gen_summary or not ref_summary:
            print(f"Skipping due to missing summary: Pretrained - {gen_summary}, Reference - {ref_summary}")
            continue

        # Compute ROUGE scores
        scores = scorer.score(ref_summary, gen_summary)

        # Append scores to respective lists
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

        precision_scores.append(scores['rougeL'].precision)
        recall_scores.append(scores['rougeL'].recall)
        f1_scores.append(scores['rougeL'].fmeasure)

    # Calculate averages
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0.0
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    # Write all metrics to a single text file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Average ROUGE-1: {avg_rouge1:.4f}\n")
        f.write(f"Average ROUGE-2: {avg_rouge2:.4f}\n")
        f.write(f"Average ROUGE-L: {avg_rougeL:.4f}\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F1-Score: {avg_f1:.4f}\n")

    print(f"ROUGE scores calculated and saved in {output_file}")

if __name__ == "__main__":
    # Input JSON files
    pretrained_summaries_path = "./data/test/test_summary.json"  # Path to test summaries
    reference_summaries_path = "./ROGUE_evaluation_pretrained/reference_summaries.json"  # Path to reference summaries

    # Saved output file
    output_file = "./ROGUE_evaluation_finetuned/rouge_scores_finetuned.txt"

    calculate_rouge_scores(pretrained_summaries_path, reference_summaries_path, output_file)
