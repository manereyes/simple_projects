from langchain_text_splitters import TokenTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from nltk.tokenize import word_tokenize
from pathlib import Path
import numpy as np
import faiss

###

FILE = Path('./mitre_attack.csv')
NEAREST_NEIGHBORS = 4
LLM = "llama3.1"
query = "Linux Reverse Shell"

###

loader = CSVLoader(file_path=FILE)
data = loader.load()

###

text_splitter = TokenTextSplitter(
    chunk_size=8000,
    chunk_overlap=2000
)

chunks = text_splitter.split_documents(documents=data)

### Combining BM25 and FAISS ###

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

###

bm25_retriever = BM25Retriever.from_documents(
    chunks,
    k=10,
    preprocess_func=word_tokenize
)

bm25_results = bm25_retriever.invoke(query)
for idx, result in enumerate(bm25_results):
    print(f"{idx} - [{result}]\n")

filtered_docs = [result for result in bm25_results]  # list comprehension of the results

filtered_vectors = embeddings.embed_documents([doc.page_content for doc in filtered_docs])  # Embed the filtered documents
filtered_vectors = np.array(filtered_vectors)  # Convert to a NumPy array
faiss_filtered_index = faiss.IndexFlatL2(len(filtered_vectors[0]))  # Create a FAISS index
faiss_filtered_index.add(filtered_vectors)  # Add the document embeddings to the index

distances, indices = faiss_filtered_index.search(filtered_vectors, NEAREST_NEIGHBORS) # Perform the search
refined_results = [filtered_docs[idx] for idx in indices[0]]

#for idx, result in enumerate(refined_results):
#    print(f"{idx} - [{result}]\n")

### LLM ###

llm = OllamaLLM(model=LLM, temperature=0.5)

template = """
You are a cybersecurity expert analyst, you will be given a security signature and a list of retrieved MITRE ATT&CK documents tactics and techniques related to it.
Your job is to generate a report about the security signature explaining the important information of the tactics and techniques related.
Consider name, ID and description of every tactic and technique.

Signature: {signature}
T&T: {ttps}

Report:


"""

prompt = PromptTemplate.from_template(
    template=template
)

formatted_prompt = prompt.format(signature=query, ttps=refined_results)
#print(formatted_prompt)

response = llm.invoke(formatted_prompt)
print(response)




