import os
import json
import html
import unicodedata
import re
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, ROOT = os.getcwd(), input_dir="json_documents", output_dir="processed_documents", min_paragraph_length=200, chunk_size=2000, chunk_overlap=200):
        self.input_dir = os.path.join(ROOT, input_dir)
        self.output_dir = os.path.join(ROOT, output_dir)
        self.chunk_size = chunk_size
        self.min_paragraph_length = min_paragraph_length
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.chunk_overlap = chunk_overlap
        self.chunker = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=2,  # Consider 2 sentences for context
            threshold=0.5,   # Higher threshold = more chunks
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
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
        
        lines = [line.strip() for line in content.split('\n')]
        lines = [line for line in lines if line]
        content = '\n'.join(lines)
        content = re.sub(r'\n{3,}', '\n\n', content)    
    
        return content
    
    def merge_paragraphs(self, cleaned_text):
        heading_pattern = re.compile(r'^(.*)\n-+$', re.MULTILINE)
        paragraphs = re.split(r'\n\n+', cleaned_text)
        
        merged_paragraphs = []
        i = 0
        
        while i < len(paragraphs):
            current = paragraphs[i].strip()
            is_heading = heading_pattern.search(current) is not None
            
            if is_heading:
                merged_paragraphs.append(current)
                i += 1
                continue
                
            merged = current
            i += 1
            
            while (i < len(paragraphs) and 
                len(merged) < self.min_paragraph_length and 
                not heading_pattern.search(paragraphs[i].strip())):
                
                next_para = paragraphs[i].strip()
                merged += "\n\n" + next_para
                i += 1
                
            if merged:
                merged_paragraphs.append(merged)
        
        return "\n\n".join(merged_paragraphs)
    
    def calculate_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def chunk_doc(self, doc):
        content = doc['content']
        metadata = doc['metadata']
        
        
        clean_content = self.clean_text(content)
        clean_content = self.merge_paragraphs(clean_content)
        
        doc = Document(page_content=clean_content, metadata = metadata)
        
        split_doc = self.chunker.split_documents([doc])
        
        chunks = []
        for i, chunk in enumerate(split_doc):
            chunk.metadata["chunk_id"] = i
            chunks.append({"page_content": chunk.page_content,
                           "metadata": chunk.metadata})
        return chunks
        
        
    # def chunk_doc(self, doc):
    #     content = doc['content']
    #     metadata = doc['metadata']
    #     logical_sections = re.split(r'\n([^\n]+)\n[-]+\n', content)
    #     #break into logical chunks which do not exceed chunK_size
    #     chunks = []
    #     for section in logical_sections:
    #         #check if section fits in chunk. if not, break into paragraphs
    #         current_text = ""
    #         if len(section) > self.chunk_size:
    #             paragraphs = re.split(r'\n\n+', section)
    #             current_text = ""
    #             current_length = 0
                
    #             for p in paragraphs:
    #                 #if paragraph doesn't fit in chunk -> finalize chunk
    #                 if current_length + len(p) > self.chunk_size and current_text != "":
    #                     chunks.append({"page_content": current_text,
    #                                    "metadata": {**metadata, "chunk_id": len(chunks)}})
                        
    #                     #reset current check based on if chunk overlap is desired
    #                     if self.chunk_overlap > 0:
    #                         overlap_start = max(0, len(current_text) - self.chunk_overlap)
    #                         overlap_text = current_text[overlap_start:]
    #                         last_p = overlap_text.rfind('\n\n') #find last paragraph break
    #                         if last_p != -1: current_text = overlap_text[last_p + 2:]
    #                         else: current_text = overlap_text
    #                     else:
    #                         current_text = ""
                            
    #                 #start current_texta s p if it's empty      
    #                 if current_text == "": current_text = p
    #                 else: current_text += "\n\n" + p
    #         else:
    #             chunks.append({"page_content": section,
    #                             "metadata": {**metadata, "chunk_id": len(chunks)}})
    #         #last chunk
    #         if current_text != "":
    #             chunk_doc = {
    #                 "page_content": current_text,
    #                 "metadata": {
    #                     **metadata,
    #                     "chunk_id": len(chunks)
    #                 }
    #             }
    #             chunks.append(chunk_doc)
        
    #     # chunk_count
    #     for chunk in chunks:
    #         chunk["metadata"]["chunk_count"] = len(chunks)
            
    #     return chunks
                        
                    
    def process_and_save_docs(self, output_filename="langchain_documents.json"):
        docs = self.load_docs()[:10]
        all_chunks = []
        for doc in docs:
            chunks = self.chunk_doc(doc)
            all_chunks.extend(chunks)
            
        filepath = os.path.join(self.output_dir, output_filename)
    
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        return all_chunks
        
def main():
    processor = DocumentProcessor()
    #saves docs to output_directory/output_filename, returns list of page_content/metadata dicts
    processed_docs = processor.process_and_save_docs(output_filename="langchain_documents.json") 
    
if __name__== "__main__":
    main()