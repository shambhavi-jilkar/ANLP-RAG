import os
import json
import pickle
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# File to cache the vector store
EMBEDDINGS_FILE = "doc_embeddings.pkl"

def build_vectorstore():
    loader = DirectoryLoader('data/documents', glob="*.txt")
    docs = loader.load()

    #text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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

def run_experiments(test_q, output_json):
    results = {}
    with open(test_q, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    for idx, question in enumerate(questions, start=1):
        print(f"Processing Question {idx}: {question}")
        answer = qa_chain.run(question)
        results[str(idx)] = answer
        print(f"Answer {idx}: {answer}\n{'-'*40}")

    with open(output_json, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)

    print(f"Experiment results saved to {output_json}")

if __name__ == "__main__":
    questions_txt = "data/test/questions.txt"   
    output_json = "system_output.json"          
    run_experiments(questions_txt, output_json)