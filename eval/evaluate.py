import os
import json
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def normalize_answer(s: str) -> str:
    def lower(text): 
        return text.lower()
    def remove_punc(text): 
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): 
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): 
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))

def compute_f1(gold: str, pred: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(pred).split()
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    common = set(gold_tokens) & set(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def match_numeric(gold: str, pred: str, tol: float = 1e-5) -> float:
    gold_clean = re.sub(r"[^\d.]", "", gold)
    pred_clean = re.sub(r"[^\d.]", "", pred)
    try:
        gold_num = float(gold_clean)
        pred_num = float(pred_clean)
        return 1.0 if abs(gold_num - pred_num) < tol else 0.0
    except ValueError:
        return 0.0

def match_year(gold: str, pred: str) -> float:
    year_pattern = r"(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})"
    gold_years = re.findall(year_pattern, gold)
    pred_years = re.findall(year_pattern, pred)
    if gold_years and pred_years:
        return 1.0 if any(y in pred_years for y in gold_years) else 0.0
    return 0.0

def match_location(gold: str, pred: str) -> float:
    return compute_f1(gold, pred)

def compute_bleu(gold: str, pred: str) -> float:
    gold_tokens = gold.lower().split()
    pred_tokens = pred.lower().split()
    if not gold_tokens or not pred_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothing)


def evaluate_row(gold_ans: str, pred_ans: str, answer_type: str) -> dict:
    gold_ans = gold_ans.strip()
    pred_ans = (pred_ans or "").strip()
    
    if not gold_ans and not pred_ans:
        return {"exact": 1.0, "f1": 1.0, "bleu": 1.0}
    if not gold_ans or not pred_ans:
        return {"exact": 0.0, "f1": 0.0, "bleu": 0.0}
    
    bleu = compute_bleu(gold_ans, pred_ans)
    exact = compute_exact(gold_ans, pred_ans)
    f1 = compute_f1(gold_ans, pred_ans)
    
    return {"exact": exact, "f1": f1, "bleu": bleu}
    
def evaluate_results(sys_outputs_path: str, ref_answers_path: str, summary_output_path: str):
    # Load system outputs and reference data
    with open(sys_outputs_path, "r", encoding="utf-8") as f:
        sys_outputs = json.load(f)
    with open(ref_answers_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    
    all_exact, all_f1, all_bleu = [], [], []
    type_metrics = {}
    output_lines = []  # to capture all evaluation output

    
    # Evaluate each question
    for qid, gold_info in ref_data.items():
        gold_ans = gold_info["answer"]
        answer_type = gold_info.get("answer_type", "text")
        pred_ans = sys_outputs.get(qid, "")
        metrics = evaluate_row(gold_ans, pred_ans, answer_type)
        all_exact.append(metrics["exact"])
        all_f1.append(metrics["f1"])
        all_bleu.append(metrics["bleu"])
        
        if answer_type not in type_metrics:
            type_metrics[answer_type] = {"exact": [], "f1": [], "bleu": []}
        type_metrics[answer_type]["exact"].append(metrics["exact"])
        type_metrics[answer_type]["f1"].append(metrics["f1"])
        type_metrics[answer_type]["bleu"].append(metrics["bleu"])
        
        line = (f"Q{qid}: Type: {answer_type}\n"
                f"      Gold: {gold_ans}\n"
                f"      Pred: {pred_ans}\n"
                f"      EM: {metrics['exact']:.2f}, F1: {metrics['f1']:.2f}, BLEU: {metrics['bleu']:.2f}\n")
        output_lines.append(line)
    
    avg_exact = sum(all_exact) / len(all_exact) if all_exact else 0.0
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    avg_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
    
    overall_summary = (
        "=== Overall Evaluation Metrics ===\n"
        f"Average Exact Match: {avg_exact:.3f}\n"
        f"Average F1 Score:      {avg_f1:.3f}\n"
        f"Average BLEU Score:    {avg_bleu:.3f}\n"
    )
    output_lines.append(overall_summary)
    
    breakdown = "=== Breakdown by Answer Type ===\n"
    for ans_type, metrics in type_metrics.items():
        type_exact = sum(metrics["exact"]) / len(metrics["exact"])
        type_f1 = sum(metrics["f1"]) / len(metrics["f1"])
        type_bleu = sum(metrics["bleu"]) / len(metrics["bleu"])
        breakdown_line = f"{ans_type:>10} -> EM: {type_exact:.3f} | F1: {type_f1:.3f} | BLEU: {type_bleu:.3f}\n" 
        breakdown += breakdown_line
    output_lines.append(breakdown)
    
    # Write the summary output to a text file
    summary_text = "\n".join(output_lines)
    with open(summary_output_path, "w", encoding="utf-8") as out_file:
        out_file.write(summary_text)

    with open(summary_output_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(output_lines))
    
    return summary_text

if __name__ == "__main__":
    sys_outputs_path1 = "cleaned_system_outputs_mistral_rag_specificOneWord.json"
    sys_outputs_path2 = "system_outputs_v1-clean.json"
    sys_outputs_path3 = "cleaned_system_outputs_mistral_rag.json"
    sys_outputs_path4 = "cleaned_system_outputs_llama_fewshot_prompt_ablation.json"
    sys_outputs_path5 = "cleaned_system_outputs_rag.json"
    ref_answers_path = "ANLP-RAG\data\test\reference_answers.json"

    summary_output_path1 = "evaluation_summary1.txt"
    summary_output_path2 = "evaluation_summary2.txt"
    summary_output_path3 = "evaluation_summary3.txt"
    summary_output_path4 = "evaluation_summary4.txt"
    summary_output_path5 = "evaluation_summary5.txt"

    summary1 = evaluate_results(sys_outputs_path1, ref_answers_path, summary_output_path1)
    summary2 = evaluate_results(sys_outputs_path2, ref_answers_path, summary_output_path2)
    summary3 = evaluate_results(sys_outputs_path3, ref_answers_path, summary_output_path3)
    summary4 = evaluate_results(sys_outputs_path4, ref_answers_path, summary_output_path4)
    summary5 = evaluate_results(sys_outputs_path5, ref_answers_path, summary_output_path5)
    print(f"Evaluation summary saved to {summary_output_path1}")
