from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import gc
import concurrent.futures
import csv
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HG_KEY")) #ADD HG_KEY TO .env file

torch.cuda.empty_cache()
gc.collect()

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = FAISS.load_local("./knowledge_base/vector_db/", embedding_function,
                              allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

model_id = "mistralai/Mistral-7B-Instruct-v0.3"  
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16, 
    device_map="auto",  
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=100, 
    temperature=0.1, 
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    batch_size=8  
)
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""<s>[INST] Based on the following context, answer this question concisely. Do not use full sentences, just directly give the answer and you can use one word answers where possible:

Context:
{context}

Question: {question} [/INST]
"""
)

def retrieve_answers_batch(questions, batch_size=32):
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        batch_contexts = list(executor.map(lambda q: '\n'.join([doc.page_content for doc in retriever.get_relevant_documents(q)]), questions))
    
    batch_prompts = [prompt_template.format(question=q, context=(batch_contexts[i])) for i, q in enumerate(questions)]
    
    batch_responses = llm.generate(batch_prompts)
    
    return [gen[0].text.strip() for gen in batch_responses.generations]


def run_model(questions_file, output_file, batch_size=32):
    with open(questions_file, "r", encoding="utf-8") as file:
        questions = file.read().splitlines()
    
    output = {}
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(questions) + batch_size - 1)//batch_size}")
        
        responses = retrieve_answers_batch(batch)
        
        for j, response in enumerate(responses):
            output[i+j+1] = response
        
        if (i + batch_size) % (batch_size * 2) == 0:
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()
            gc.collect()
        
        if (i + batch_size) % (batch_size * 3) == 0:
            print(f"Saving intermediate results at question {i+batch_size}...")
            with open(f"{output_file}.partial", "w", encoding="utf-8") as file:
                json.dump(output, file, indent=4, ensure_ascii=False)
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)
    
    return output


QUESTIONS_FILE = "./data/test/questions.txt"
OUTPUT_FILE = "./data/system_outputs/system_outputs_mistral_rag_specificOneWord.json"

output = run_model(questions_file=QUESTIONS_FILE, output_file=OUTPUT_FILE, batch_size=32)