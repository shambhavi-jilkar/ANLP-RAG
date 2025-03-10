import os
import json
import pickle
import wandb
import re
import string
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Load environment variables
load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
if api_key:
    wandb.login(key=api_key)
else:
    print("WANDB_API_KEY not set. Please check your .env file.")

# Initialize wandb
run = wandb.init(name="Combined_Run", reinit=True, project="anlp-rag-evaluations")

# File to cache the vector store
EMBEDDINGS_FILE = "doc_embeddings.pkl"

def build_vectorstore():
    loader = DirectoryLoader('data/documents', glob="*.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

if os.path.exists(EMBEDDINGS_FILE):
    print("Loading precomputed embeddings")
    with open(EMBEDDINGS_FILE, "rb") as f:
        vectorstore = pickle.load(f)
else:
    print("Building vector store.")
    vectorstore = build_vectorstore()
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(vectorstore, f)
    print("Embeddings computed and saved.")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
generator = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=generator)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def run_experiments(questions_txt_path, output_json_path):
    results = {}
    with open(questions_txt_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    for idx, question in enumerate(questions, start=1):
        print(f"Processing Question {idx}: {question}")
        answer = qa_chain.run(question)
        results[str(idx)] = answer
        print(f"Answer {idx}: {answer}\n{'-'*40}")
    with open(output_json_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
    print(f"Experiment results saved to {output_json_path}")
    return results

def normalize_answer(s):
    def lower(text): return text.lower()
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

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

def evaluate_results(sys_outputs_path, ref_answers_path):
    with open(sys_outputs_path, "r", encoding="utf-8") as f:
        sys_outputs = json.load(f)
    with open(ref_answers_path, "r", encoding="utf-8") as f:
        ref_answers = json.load(f)
    q_nums, exact_scores, f1_scores = [], [], []
    for q_num, gold_answer in ref_answers.items():
        if q_num in sys_outputs:
            pred_answer = sys_outputs[q_num]
            exact = compute_exact(gold_answer, pred_answer)
            f1 = compute_f1(gold_answer, pred_answer)
            q_nums.append(int(q_num))
            exact_scores.append(exact)
            f1_scores.append(f1)
            wandb.log({"Question Number": int(q_num), "Exact Match": exact, "F1 Score": f1})
            print(f"Q{q_num}: Gold: {gold_answer} | Pred: {pred_answer} | EM: {exact}, F1: {f1:.2f}")
        else:
            print(f"Question {q_num} not found in system output.")
    avg_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    wandb.log({"Average Exact Match": avg_exact, "Average F1 Score": avg_f1})
    print(f"\nAverage Exact Match: {avg_exact:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")

if __name__ == "__main__":
    # Run experiments to generate system outputs
    questions_txt = "data/test/questions.txt"
    output_json = "data/system_outputs/system_output.json"
    sys_outputs = run_experiments(questions_txt, output_json)
    
    # Evaluate results against reference answers
    ref_answers_path = "data/test/reference_answers.json"
    evaluate_results(output_json, ref_answers_path)
