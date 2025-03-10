# main.py

# Step 1: Import necessary modules
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# Step 2: Load your documents
# This will load all .txt files from the specified directory.
loader = DirectoryLoader('data/documents', glob="*.txt")
docs = loader.load()

# Step 3: Split documents into manageable chunks
# This helps the model by keeping each chunk within a reasonable size.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Step 4: Create document embeddings using a Hugging Face model.
# "all-MiniLM-L6-v2" is a lightweight, effective model for embeddings.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 5: Create a vector store with FAISS from your document chunks
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Step 6: Set up a retriever to fetch the top 3 relevant document chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Step 7: Set up your QA model using a Hugging Face pipeline.
# We'll use "google/flan-t5-small" as an example text generation model.
generator = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=generator)

# Step 8: Build the RetrievalQA chain combining the retriever and QA model.
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Step 9: Run a sample query
if __name__ == "__main__":
    query = "Who is Pittsburgh named after?"
    answer = qa_chain.run(query)
    print("Query:", query)
    print("Answer:", answer)
