import json
import wandb
import re
import string

wandb.login(key="22a4d9a6fe8ff8bf1fabc55360f865d49b5f26d9")
run = wandb.init(
    name    = "Try1", ### Wandb creates random run names if you skip this field, we recommend you give useful names
    reinit  = True, ### Allows reinitalizing runs when you re-run this cell
    project = "anlp-rag-evaluations" ### Project should be created in your wandb account
)

with open("data/system_outputs/system_output.json", "r", encoding="utf-8") as f:
    sys_outputs = json.load(f)

with open("data/test/reference_answers.json", "r", encoding="utf-8") as f:
    ref_answers = json.load(f)

def normalize_answer(s):
    def lower(text):
        return text.lower()
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_tokens = normalize_answer(a_gold).split()
    pred_tokens = normalize_answer(a_pred).split()
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return int(gold_tokens == pred_tokens)
    common_tokens = set(gold_tokens) & set(pred_tokens)
    if not common_tokens:
        return 0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

q_nums = []
exact_scores = []
f1_scores = []
labels = []

for q_num, gold_answer in ref_answers.items():
    if q_num in sys_outputs:
        pred_answer = sys_outputs[q_num]
        exact = compute_exact(gold_answer, pred_answer)
        f1 = compute_f1(gold_answer, pred_answer)
        q_nums.append(int(q_num))
        exact_scores.append(exact)
        f1_scores.append(f1)
        labels.append(f"Q{q_num}")
    else:
        print(f"Question {q_num} not found in system output.")

# Calculate average scores
avg_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0
avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

print(f"Average Exact Match: {avg_exact:.2f}")
print(f"Average F1 Score: {avg_f1:.2f}")

for q, em, f1 in zip(q_nums, exact_scores, f1_scores):
    wandb.log({
        "Question Number": q,
        "Exact Match": em,
        "F1 Score": f1
    })
    
wandb.log({
    "Average Exact Match": avg_exact,
    "Average F1 Score": avg_f1
})
print(f"\nAverage Exact Match: {avg_exact:.2f}")
print(f"Average F1 Score: {avg_f1:.2f}")
