from langchain.llms import HuggingFacePipeline
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import csv
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os
import gc

load_dotenv()
login(token=os.getenv("HG_KEY")) #ADD HG_KEY TO .env file


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("./knowledge_base/vector_db/",embedding_function,
                               allow_dangerous_deserialization=True)

print(type(vectorstore))
print(vectorstore.index.ntotal)


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

torch.cuda.empty_cache()
gc.collect()

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,  
    legacy=True,     
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  
    device_map="auto",
    load_in_8bit=True,  
)

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=750,
    temperature=0.1, 
    top_p=0.95, 
    repetition_penalty=1.15,
    batch_size=4 
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

llm = HuggingFacePipeline(pipeline=pipe)

examples = [{"question": "When is Picklesburgh?",
             "context": "Related Stories & Events\nPicklesburgh 2025\nJuly 11 to July 13\nThis is kind of a big \"dill.\" Picklesburgh is back and better than ever. From July 11 through July 13, pickle enthusiasts can experience the fermented fun in Downtown Pittsburgh.\n'The Thaw' Event Series Turns Up The Heat in Pittsburgh's Market Square\nPittsburgh's coolest way to warm up.\nPeoples Gas Holiday MarketTM\nGet into the holiday spirit at the 13th Annual Peoples Gas Holiday MarketTM!\nPittsburgh Holiday Lights\nExperience the sparkle this 2024 holiday season in Pittsburgh!\nPittsburgh Celebrates Light Up Night 2024\nJoin us as we light up the city and officially kick-off the Pittsburgh holiday season!",
             "answer": "July 11th to July 13th"},
            
            {"question": "Which neighborhood are the Carnegie Museums of Art and Natural History located?",
             "context": "Carnegie Museum of Art and Natural History\nExplore two world-class museums with one ticket at Carnegie Museums of Art and Natural History! Located in the Oakland neighborhood, the museums provide an experience filled with culture, creativity, and exploration to all who visit. Carnegie Museum of Art, renowned for its collections of more than 100,00 objects, features a range of visual art, including painting, sculpture, decorative arts, design, and photography. Located within the same historic building, Carnegie Museum of Natural History is home to one of the country's largest and most celebrated natural history collections featuring 22 million objects and specimens, ranging from real dinosaur fossils and dazzling gems and minerals to iconic wildlife dioramas.",
             "answer": "Oakland"},
            
            {"question": "What is the Strip District known for?",
             "context": "Strip District\nDon't be fooled by the name.\nThe Strip, as it's called, is foodie heaven and as authentic as it is fun. Locals love it for its low, low prices and tremendous selections. The one-half square mile shopping district is chock full of ethnic grocers, produce stands, meat and fish markets and sidewalk vendors. Breathe deep because you won't want to escape the splendid aromas of fresh-roasted coffee or just-baked bread. Bordering Downtown, this neighborhood is pure Pittsburgh.\n\nThe Name\n\"The Strip,\" as locals call it, is just that – a narrow strip (one-half mile) of land between the Allegheny River and a mountain of a hill.\n\nOne More Thing\nAfter building his first factory in the Strip District in 1871, George Westinghouse not only invented air brakes and AC current, but introduced paid vacations and half-days off on Saturdays.",
             "answer": "Ethnic food stalls, grocery stores, and restaurants with low prices and great variety."},
            
            {"question": "Why is the Strip District named as it is?",
             "context": "Strip District\nDon't be fooled by the name.\nThe Strip, as it's called, is foodie heaven and as authentic as it is fun. Locals love it for its low, low prices and tremendous selections. The one-half square mile shopping district is chock full of ethnic grocers, produce stands, meat and fish markets and sidewalk vendors. Breathe deep because you won't want to escape the splendid aromas of fresh-roasted coffee or just-baked bread. Bordering Downtown, this neighborhood is pure Pittsburgh.\n\nThe Name\n\"The Strip,\" as locals call it, is just that – a narrow strip (one-half mile) of land between the Allegheny River and a mountain of a hill.\n\nOne More Thing\nAfter building his first factory in the Strip District in 1871, George Westinghouse not only invented air brakes and AC current, but introduced paid vacations and half-days off on Saturdays.",
             "answer": "It is a strip of land between a river and a large hill."}]


example_template = """Question: {question}
Context: {context}
Answer: {answer}[EOS]"""

example_prompt = PromptTemplate(input_variables=['question', 'context', 'answer'], template=example_template)


few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix="""# Instructions
Answer questions directly based on the provided context.
Be concise and don't answer with full sentences.
Always end your response with [EOS] to indicate completion.
Do not add any additional text after your answer.

Here are examples of different types of questions and how to answer them with the given context:""",
    suffix="""Question: {question}
Context: {context}
Answer: """,
    input_variables=['question', 'context'],
    example_separator='\n\n')

def retrieve_answers_batch(questions, batch_size=4):
    all_responses = []
    all_contexts = []
    
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        batch_docs = []
        batch_contexts = []
        
        for q in batch:
            docs = retriever.get_relevant_documents(q)
            context = '\n\n'.join([doc.page_content for doc in docs])
            batch_docs.append(docs)
            batch_contexts.append(context)
        
        
        batch_prompts = []
        for j, q in enumerate(batch):
            prompt = few_shot_prompt.format(question=q, context=batch_contexts[j])
            batch_prompts.append(prompt)
            
        batch_responses = llm.generate(batch_prompts)
        
        for gen in batch_responses.generations:
            all_responses.append(gen[0].text)
        
        all_contexts.extend(batch_contexts)
    
    return all_responses, all_contexts

def run_model(questions_file, output_file, batch_size=8):
    with open(questions_file, "r", encoding="utf-8") as file:
        questions = file.read().splitlines()
    # with open('system_outputs_llama_fewshot.json.partial', 'r', encoding = 'utf-8') as f:
    #     answers = json.load(f)

    # done_answer_ids = set(int(key) if key.isdigit() else key for key in answers.keys())
    
    # questions = []
    # remaining_indices = []
    
    # for i, question in enumerate(all_questions):
    #     if i+1 not in done_answer_ids and str(i+1) not in done_answer_ids:
    #         questions.append(question)
    
    # output = answers
    output = {}

    for i in range(0,len(questions), batch_size):
        batch = questions[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(questions) + batch_size - 1)//batch_size}")
        
        responses, contexts = retrieve_answers_batch(batch)
        
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


## single doc version
# def retreive_answer(q):
#     docs = retriever.get_relevant_documents(q)
#     context = '\n\n'.join([doc.page_content for doc in docs])
#     prompt = few_shot_prompt.format(question=q, context=context)
#     response = llm(prompt)
#     return response, context


QUESTIONS_FILE = "./data/test/questions.txt"
OUTPUT_FILE = "./data/system_outputs/system_outputs_llama_fewshot.json"

output = run_model(questions_file=QUESTIONS_FILE, output_file=OUTPUT_FILE, batch_size=8)
