# CMU Advanced NLP Assignment 2: RAG System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for question-answering on topics related to Pittsburgh and CMU. The system uses LangChain along with FAISS and Transformers to retrieve relevant documents and generate answers.

## Setup

### 1. Create a Virtual Environment
python3 -m venv anlp-hw2

### 2. Activate the Virtual Environment
On Windows:
anlp-hw2\Scripts\activate
On macOS/Linux:
source anlp-hw2/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Create a .env file with your huggingface token
HGKEY="enter your key"

### 5. Run Project
python3 main.py 
python3 evaluate.py

