import os
import json
import html
import unicodedata
import re
import numpy as np
import pickle
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class DocumentProcessor:
    def __init__(self, ROOT = os.getcwd(), input_dir="json_documents", output_dir="processed_documents", min_paragraph_length=200, chunk_size=5000):
        self.input_dir = os.path.join(ROOT, input_dir)
        self.output_dir = os.path.join(ROOT, output_dir)
        self.chunk_size = chunk_size
        self.min_paragraph_length = min_paragraph_length
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        # self.chunker = SemanticChunker(
        #     embeddings=self.embeddings,
        #     buffer_size=2,  # Consider 2 sentences for context
        #     breakpoint_threshold_amount=0.5   # Higher threshold = more chunks
        # )
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_docs(self):
        docs = []
        for filename in os.listdir(self.input_dir):
            if not filename.endswith('.json') : continue
            filepath = os.path.join(self.input_dir, filename)
            with open(filepath, 'r', encoding = 'utf-8') as f:
                doc = json.load(f)
                docs.append(doc)
        return docs
    
    def clean_text(self, content):
        content = html.unescape(content)
        content = unicodedata.normalize('NFKC', content)
        content = re.sub(r' +', ' ', content)
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', content)
        
        content = re.sub(r'https?://\S+', '', content)
        content = re.sub(r'www\.\S+', '', content)
        
        lines = [line.strip() for line in content.split('\n')]
        lines = [line for line in lines if line]
        content = '\n'.join(lines)
        content = re.sub(r'\n{3,}', '\n\n', content)    
    
        return content
    
    #semantic implementaiton - it's not working as well as using headers    
    # def chunk_doc(self, doc):
    #     content = doc['content']
    #     metadata = doc['metadata']
        
        
    #     clean_content = self.clean_text(content)
    #     clean_content = self.merge_paragraphs(clean_content)
        
    #     doc = Document(page_content=clean_content, metadata = metadata)
        
    #     split_doc = self.chunker.split_documents([doc])
        
    #     chunks = []
    #     for i, chunk in enumerate(split_doc):
    #         chunk.metadata["chunk_id"] = i
    #         chunks.append({"page_content": chunk.page_content,
    #                        "metadata": chunk.metadata})
    #     return chunks
        
        
    def chunk_doc(self, doc):
        content = doc['content']
        metadata = doc['metadata']
        
        cleaned_text = self.clean_text(content)
        
        lines = cleaned_text.split('\n')[2:]
        chunks = []
        i = 0
        skip_section = False
        
        while i < len(lines):
            line = lines[i].strip()
            
            #check for header line
            if i + 1 < len(lines) and re.match(r'^-+$', lines[i+1].strip()):
                skip_section = line.lower() in ["contents", "see also", "references", "external links"]
                if skip_section: #skip unecessary sections
                    i += 2  #skip two lines
                    while i < len(lines) and not (i + 1 < len(lines) and re.match(r'^-+$', lines[i+1].strip())): #iterate through to next header
                        i += 1
                    continue
                
                header = line
                i += 2  #skip 2 lines
                
                section_content = []
                while i < len(lines) and not (i + 1 < len(lines) and re.match(r'^-+$', lines[i+1].strip())): #collect content until next header
                    section_content.append(lines[i])
                    i += 1
                
                #skip empty
                if not section_content:
                    continue
                
                #attach to header
                section_text = header + "\n" + "\n".join(section_content)
                if len(section_text) <= self.chunk_size: #check if it fits
                    chunks.append({
                        "page_content": section_text,
                        "metadata": {**metadata, "chunk_id": len(chunks)}
                    })
                else:
                    # split into paragraphs
                    paragraphs = re.split(r'\n\n+', "\n".join(section_content))
                    current_chunk = header
                    
                    for para in paragraphs:
                        if len(current_chunk) + len(para)> self.chunk_size and current_chunk != header:
                            chunks.append({
                                "page_content": current_chunk,
                                "metadata": {**metadata, "chunk_id": len(chunks)}
                            })
                            current_chunk = header + "\n" + para
                        else:
                            current_chunk += "\n\n" + para
                    
                    # add in any trailing content
                    if current_chunk and current_chunk != header:
                        chunks.append({
                            "page_content": current_chunk,
                            "metadata": {**metadata, "chunk_id": len(chunks)}
                        })
            else:#check for any content before header
                if not skip_section:
                    pre_header_content = []
                    while i < len(lines) and not (i + 1 < len(lines) and re.match(r'^-+$', lines[i+1].strip())):
                        if lines[i].strip():
                            pre_header_content.append(lines[i].strip())
                        i += 1
                    
                    if pre_header_content:
                        content_text = "\n".join(pre_header_content)
                        chunks.append({
                            "page_content": content_text,
                            "metadata": {**metadata, "chunk_id": len(chunks)}
                        })
                else:
                    i += 1
        
        for chunk in chunks:
            chunk["metadata"]["chunk_count"] = len(chunks)
        
        return chunks
                                
                        
    def process_and_save_docs(self, output_filename="langchain_documents.json"):
        docs = self.load_docs()
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_doc(doc)
            all_chunks.extend(chunks)
            
        filepath = os.path.join(self.output_dir, output_filename)
    
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        return all_chunks
    
def build_vectorstore(chunks, output_file, new_docs_only = False):
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(output_file) and new_docs_only: ##UNIQUE DOCS ONLY IS WIP FUNCTIONALITY - HAVE TO FIND HOW TO EFFICIENTLY EXTRACT METADATA
        print("Loading existing embeddings")
        with open(output_file, "rb") as f:
            vectorstore = pickle.load(f)
        print("adding new docs")
        existing_docs = vectorstore.get() 
        existing_urls = set()
        
        for doc in existing_docs['metadatas']:
            if 'source' in doc:
                existing_urls.add(doc['source'])
        new_docs = []
        if chunk['metadata']['url'] not in existing_urls:
            new_docs.append(Document(
                page_content=chunk['page_content'],
                metadata=chunk['metadata']
            ))
            
        vectorstore.add_documents(new_docs)
        with open(output_file, "wb") as f:
            pickle.dump(vectorstore, f)
        print("New doc embeddings added & saved to vector db")
        
    else:
        print("Building vector store.")
        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"]
            )
            docs.append(doc)
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(output_file, "wb") as f:
            pickle.dump(vectorstore, f)
        print("Embeddings computed and saved.")
        
    return vectorstore
        
def main():
    processor = DocumentProcessor()
    #saves docs to output_directory/output_filename, returns list of page_content/metadata dicts
    processed_docs = processor.process_and_save_docs(output_filename="langchain_documents.json") 
    print(len(processed_docs))
    
    EMBEDDINGS_FILE = "doc_embeddings.pkl"
    vec_db = build_vectorstore(processed_docs, EMBEDDINGS_FILE, new_docs_only=False)
    

    
if __name__== "__main__":
    main()