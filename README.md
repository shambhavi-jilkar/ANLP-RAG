# CMU Advanced NLP Assignment 2: RAG System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for question-answering on topics related to Pittsburgh and CMU. The system uses LangChain along with FAISS and Transformers to retrieve relevant documents and generate answers.

## Setup

### 1. Create a Virtual Environment
python3.10 -m venv myenv

### 2. Activate the Virtual Environment
On Windows:
myenv\Scripts\activate
On macOS/Linux:
source myenv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Create a .env file with your huggingface token
HGKEY="enter your key"

### 5. Documents processor
knowledge_base/json_documents contains a list of raw scraped documents that are samples. Add all extra scraped documents as needed. (Scraper scripts have been added, modify based on websites)
Run 'python DocumentProcessor.py' to create the vector_db
Complete Knowledge Base with VectorDB is located at https://drive.google.com/drive/folders/1yrjnwG0OKfTJVhPessC858rNd62pUnBu?usp=drive_link. (Optionally download if needed)

### 6. Pipeline 
Ensure that vector_db is present in knowledge_base before running. 
(Change directory paths for input and output file as needed)
python Mistral_7B_StandardRag_OneWord.py

### 7. Evaluation
Takes in model prediction outputs and reference answers to compute metrics for evaluation
(Change directory paths for input and output file as needed)
python eval/evaluate.py

