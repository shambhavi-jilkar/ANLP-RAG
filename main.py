import os
import json
import torch
import pickle
from huggingface_hub import login
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()
# Log in to Hugging Face Hub (ensure you trust the source)
login(token=os.getenv("HG_KEY"))

# Load the embedding function and pre-built vector store
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Note: adjust the path to your vector store directory (it was saved using FAISS.save_local)
vectorstore = FAISS.load_local("vector_db/", embedding_function, allow_dangerous_deserialization=True)
print(f"Loaded vector store with {vectorstore.index.ntotal} documents.")
# Create a retriever that will fetch the top 4 relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Set up few-shot examples (these should be representative of the kinds of questions you'll ask)
examples = [
    {
        "question": "When is Picklesburgh?",
        "context": "Related Stories & Events\nPicklesburgh 2025\nJuly 11 to July 13\nThis is kind of a big 'dill.' Picklesburgh is back and better than ever.",
        "answer": "July 11th to July 13th"
    },
    {
        "question": "Which neighborhood are the Carnegie Museums of Art and Natural History located?",
        "context": "Carnegie Museum of Art and Natural History\nExplore two world-class museums located in Oakland, Pittsburgh.",
        "answer": "Oakland"
    },
    {
        "question": "What is the Strip District known for?",
        "context": "Strip District\nThe one-half square mile shopping district is full of ethnic grocers, produce stands, and restaurants offering low prices and great variety.",
        "answer": "Ethnic food stalls, grocery stores, and restaurants with low prices and great variety."
    },
    {
        "question": "Why is the Strip District named as it is?",
        "context": "Strip District\nLocals call it 'the Strip' because it is a narrow strip of land between the Allegheny River and a large hill.",
        "answer": "It is a strip of land between a river and a large hill."
    }
]

# Create an example prompt template
example_template = """Question: {question} Context: {context} Answer: {answer}"""
example_prompt = PromptTemplate(input_variables=['question', 'context', 'answer'], template=example_template)
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix="Answer questions based on the provided context. Here are some examples:",
    suffix="Question: {question} Context: {context} Answer:",
    input_variables=['question', 'context'],
    example_separator="\n\n"
)

# Set up the LLM using a HuggingFace pipeline
model_id = "mistralai/Mistral-Small-24B-Instruct-2501"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=pipe)

def retrieve_answer(question: str):
    # Retrieve relevant documents for the given question
    docs = retriever.get_relevant_documents(question)
    # Concatenate the retrieved documents' content to form the context
    context = "\n\n".join([doc.page_content for doc in docs])
    # Format the few-shot prompt with the current question and retrieved context
    prompt = few_shot_prompt.format(question=question, context=context)
    # Generate the answer using the LLM
    response = llm(prompt)
    return response, context

def run_experiments(questions_txt_path, output_json_path):
    results = {}
    with open(questions_txt_path, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    for idx, question in enumerate(questions, start=1):
        print(f"Processing Question {idx}: {question}")
        answer, context = retrieve_answer(question)
        results[str(idx)] = answer
        print(f"Answer {idx}: {answer}\n{'-'*40}")
    with open(output_json_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)
    print(f"Experiment results saved to {output_json_path}")
    return results

if __name__ == "__main__":
    questions_txt = "data/test/questions.txt"  # Make sure this file contains your questions (one per line)
    output_json = "data/test/system_output.json"
    run_experiments(questions_txt, output_json)
