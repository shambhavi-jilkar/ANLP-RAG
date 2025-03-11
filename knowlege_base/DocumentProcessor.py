import os
import json
import re

class DocumentProcessor:
    def __init__(self, ROOT = os.getcwd(), input_dir="json_documents", output_dir="processed_documents", chunk_size=1000, chunk_overlap=200):
        self.input_dir = os.path.join(ROOT, input_dir)
        self.output_dir = os.path.join(ROOT, output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
    
    def chunk_doc(self, doc):
        content = doc['content']
        metadata = doc['metadata']
        logical_sections = re.split(r'\n([^\n]+)\n[-]+\n', content)
        #break into logical chunks which do not exceed chunK_size
        chunks = []
        for section in logical_sections:
            #check if section fits in chunk. if not, break into paragraphs
            current_text = ""
            if len(section) > self.chunk_size:
                paragraphs = re.split(r'\n\n+', section)
                current_text = ""
                current_length = 0
                
                for p in paragraphs:
                    #if paragraph doesn't fit in chunk -> finalize chunk
                    if current_length + len(p) > self.chunk_size and current_text != "":
                        chunks.append({"page_content": current_text,
                                       "metadata": {**metadata, "chunk_id": len(chunks)}})
                        
                        #reset current check based on if chunk overlap is desired
                        if self.chunk_overlap > 0:
                            overlap_start = max(0, len(current_text) - self.chunk_overlap)
                            overlap_text = current_text[overlap_start:]
                            last_p = overlap_text.rfind('\n\n') #find last paragraph break
                            if last_p != -1: current_text = overlap_text[last_p + 2:]
                            else: current_text = overlap_text
                        else:
                            current_text = ""
                            
                    #start current_texta s p if it's empty      
                    if current_text == "": current_text = p
                    else: current_text += "\n\n" + p
            else:
                chunks.append({"page_content": section,
                                "metadata": {**metadata, "chunk_id": len(chunks)}})
            #last chunk
            if current_text != "":
                chunk_doc = {
                    "page_content": current_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": len(chunks)
                    }
                }
                chunks.append(chunk_doc)
        
        # chunk_count
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
        
def main():
    processor = DocumentProcessor()
    #saves docs to output_directory/output_filename, returns list of page_content/metadata dicts
    processed_docs = processor.process_and_save_docs(output_filename="langchain_documents.json") 
    
if __name__== "__main__":
    main()