
import os
os.environ['HF_HOME'] = "/local/athanasiadisc/cache"
from timeit import default_timer as timer

import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm

def time_tick():
    tick = timer()
    return tick

def duration2tcks(tick1, tick2):
    return (tick2 - tick1)

f = open('../../_files/rag_text.txt')
texts = f.readlines()
# Paste your API key here. Remember to not share publicly
api_key = 'M2GKPBFIZeVVrDI447YcZ5fiWVlJ3dtTilCb9861'

# Create and retrieve a Cohere API key from os.cohere.ai
co = cohere.Client(api_key)

# with open('../../_files/rag_text.txt') as text:
#   s = " ".join([l.rstrip("\n") for l in text]) 

# # Split into a list of sentences
# texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]

# Get the embeddings
response = co.embed(
  texts=texts,
  input_type="search_document",
).embeddings

embeds = np.array(response)
print(embeds.shape)

import faiss
dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))

def search(query, number_of_results=3):
  
  # 1. Get the query's embedding
  query_embed = co.embed(texts=[query], 
                input_type="search_query",).embeddings[0]

  # 2. Retrieve the nearest neighbors
  distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results) 

  # 3. Format the results
  texts_np = np.array(texts) # Convert texts list to numpy for easier indexing
  results = pd.DataFrame(data={'texts': texts_np[similar_item_ids[0]], 
                              'distance': distances[0]})
  
  # 4. Print and return the results
  print(f"Query:'{query}'\nNearest neighbors:")
  return results


query = "What is the mass of the moon?"
results = search(query)
print (results)

from langchain import LlamaCpp

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="../../_files/Phi-3-mini-4k-instruct-fp16.gguf",
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Embedding model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='thenlper/gte-small'
)


from langchain.vectorstores import FAISS

# Create a local vector database
db = FAISS.from_texts(texts, embedding_model)

from langchain import PromptTemplate

# Create a prompt template
template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


from langchain.chains import RetrievalQA

# RAG pipeline
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt
    },
    verbose=True
)

start = timer()
temp = rag.invoke('Who did the  storyboarding and screenwriting?')
end = timer()
print(end - start)
