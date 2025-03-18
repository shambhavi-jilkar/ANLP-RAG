import os
import json
import re
import string
from bert_score import score as bert_score
import matplotlib.pyplot as plt
import numpy as np

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


def compute_bertscore(gold: str, pred: str) -> float:
    P, R, F1 = bert_score([pred], [gold], lang="en", verbose=False)
    return F1[0].item()


def evaluate_row(gold_ans: str, pred_ans: str, answer_type: str) -> dict:
    gold_ans = gold_ans.strip()
    pred_ans = (pred_ans or "").strip()
    
    if not gold_ans and not pred_ans:
        return {"exact": 1.0, "f1": 1.0, "bert_f1": 1.0}
    if not gold_ans or not pred_ans:
        return {"exact": 0.0, "f1": 0.0, "bert_f1": 0.0}
    
    if answer_type.lower() == "number":
        score = match_numeric(gold_ans, pred_ans)
        bert = compute_bertscore(gold_ans, pred_ans)
        return {"exact": score, "f1": score, "bert_f1": bert}
    elif answer_type.lower() == "date":
        score = match_year(gold_ans, pred_ans)
        bert = compute_bertscore(gold_ans, pred_ans)
        return {"exact": score, "f1": score, "bert_f1": bert}
    elif answer_type.lower() == "location":
        f1 = match_location(gold_ans, pred_ans)
        exact = 1.0 if f1 == 1.0 else 0.0
        bert = compute_bertscore(gold_ans, pred_ans)
        return {"exact": exact, "f1": f1, "bert_f1": bert}
    else:
        exact = compute_exact(gold_ans, pred_ans)
        f1 = compute_f1(gold_ans, pred_ans)
        bert = compute_bertscore(gold_ans, pred_ans)
        return {"exact": float(exact), "f1": f1, "bert_f1": bert}

def evaluate_results(sys_outputs_path: str, ref_answers_path: str, summary_output_path: str):
    # Load system outputs and reference data
    with open(sys_outputs_path, "r", encoding="utf-8") as f:
        sys_outputs = json.load(f)
    with open(ref_answers_path, "r", encoding="utf-8") as f:
        ref_data = json.load(f)
    
    all_exact, all_f1, all_bert = [], [], []
    type_metrics = {}
    output_lines = []  # to capture all evaluation output
    
    # Evaluate each question
    for qid, gold_info in ref_data.items():
        gold_ans = gold_info["Answer"]
        answer_type = gold_info.get("Answer Datatype", "text")
        pred_ans = sys_outputs.get(qid, "")
        metrics = evaluate_row(gold_ans, pred_ans, answer_type)
        all_exact.append(metrics["exact"])
        all_f1.append(metrics["f1"])
        all_bert.append(metrics["bert_f1"])
        
        if answer_type not in type_metrics:
            type_metrics[answer_type] = {"exact": [], "f1": [], "bert_f1": []}
        type_metrics[answer_type]["exact"].append(metrics["exact"])
        type_metrics[answer_type]["f1"].append(metrics["f1"])
        type_metrics[answer_type]["bert_f1"].append(metrics["bert_f1"])
        
        line = (f"Q{qid}: Type: {answer_type}\n"
                f"      Gold: {gold_ans}\n"
                f"      Pred: {pred_ans}\n"
                f"      EM: {metrics['exact']:.2f}, F1: {metrics['f1']:.2f}, BERTScore F1: {metrics['bert_f1']:.2f}\n")
        output_lines.append(line)
    
    avg_exact = sum(all_exact) / len(all_exact) if all_exact else 0.0
    avg_f1 = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    avg_bert = sum(all_bert) / len(all_bert) if all_bert else 0.0
    
    overall_summary = (
        "=== Overall Evaluation Metrics ===\n"
        f"Average Exact Match: {avg_exact:.3f}\n"
        f"Average F1 Score:      {avg_f1:.3f}\n"
        f"Average BERTScore F1:  {avg_bert:.3f}\n"
    )
    output_lines.append(overall_summary)
    
    breakdown = "=== Breakdown by Answer Type ===\n"
    for ans_type, metrics in type_metrics.items():
        type_exact = sum(metrics["exact"]) / len(metrics["exact"])
        type_f1 = sum(metrics["f1"]) / len(metrics["f1"])
        type_bert = sum(metrics["bert_f1"]) / len(metrics["bert_f1"])
        breakdown_line = f"{ans_type:>10} -> EM: {type_exact:.3f} | F1: {type_f1:.3f} | BERTScore F1: {type_bert:.3f}\n"
        breakdown += breakdown_line
    output_lines.append(breakdown)
    
    # Write the summary output to a text file
    summary_text = "\n".join(output_lines)
    with open(summary_output_path, "w", encoding="utf-8") as out_file:
        out_file.write(summary_text)
    
    # Generate visualizations and save them as images
    plot_overall_metrics(avg_exact, avg_f1, avg_bert, "data/val-try/overall_metrics.png")
    plot_breakdown_by_answer_type(type_metrics, "data/val-try/breakdown_by_answer_type.png")
    
    # Also append file paths for visualizations to the summary
    viz_info = ("\nVisualizations saved as:\n"
                " - Overall Metrics: data/val-try/overall_metrics.png\n"
                " - Breakdown by Answer Type: data/val-try/breakdown_by_answer_type.png\n")
    output_lines.append(viz_info)
    with open(summary_output_path, "w", encoding="utf-8") as out_file:
        out_file.write("\n".join(output_lines))
    
    return summary_text

def plot_overall_metrics(avg_exact, avg_f1, avg_bert, save_path="overall_metrics.png"):
    metrics = [avg_exact, avg_f1, avg_bert]
    labels = ["Exact Match", "F1 Score", "BERTScore F1"]
    
    plt.figure(figsize=(8,6))
    bars = plt.bar(labels, metrics, color=["skyblue", "orange", "green"])
    plt.ylim(0, 1)
    plt.title("Overall Evaluation Metrics")
    for bar, metric in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, f"{metric:.2f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_breakdown_by_answer_type(type_metrics, save_path="breakdown_by_answer_type.png"):
    answer_types = list(type_metrics.keys())
    avg_exacts = [sum(metrics["exact"]) / len(metrics["exact"]) for metrics in type_metrics.values()]
    avg_f1s = [sum(metrics["f1"]) / len(metrics["f1"]) for metrics in type_metrics.values()]
    avg_berts = [sum(metrics["bert_f1"]) / len(metrics["bert_f1"]) for metrics in type_metrics.values()]
    
    x = np.arange(len(answer_types))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, avg_exacts, width, label="Exact Match", color="skyblue")
    plt.bar(x, avg_f1s, width, label="F1 Score", color="orange")
    plt.bar(x + width, avg_berts, width, label="BERTScore F1", color="green")
    plt.ylim(0, 1)
    plt.xticks(x, answer_types, rotation=45, fontsize=10)
    plt.title("Breakdown of Evaluation Metrics by Answer Type", fontsize=14)
    plt.legend(fontsize=12)
    for i in range(len(answer_types)):
        plt.text(x[i]-width, avg_exacts[i]+0.02, f"{avg_exacts[i]:.2f}", ha="center", fontsize=10)
        plt.text(x[i], avg_f1s[i]+0.02, f"{avg_f1s[i]:.2f}", ha="center", fontsize=10)
        plt.text(x[i]+width, avg_berts[i]+0.02, f"{avg_berts[i]:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    sys_outputs_path = "data/val-try/system_output.json"
    ref_answers_path = "data/val-try/reference_answers.json"
    summary_output_path = "data/val-try/evaluation_summary.txt"
    summary = evaluate_results(sys_outputs_path, ref_answers_path, summary_output_path)
    print(f"Evaluation summary saved to {summary_output_path}")
