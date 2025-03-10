from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

loader = DirectoryLoader('data/documents', glob="*.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(split_docs, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

generator = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=generator)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

if __name__ == "__main__":
    query = "Who is Pittsburgh named after?"
    answer = qa_chain.run(query)
    print("Query:", query)
    print("Answer:", answer)
