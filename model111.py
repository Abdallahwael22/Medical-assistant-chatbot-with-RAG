!pip install streamlit langchain langchain-community
!pip install sentence-transformers
!pip install chromadb
!pip install transformers
!pip install torch torchvision torchaudio
!pip install huggingface_hub
!pip install pypdf
!pip install bitsandbytes

import os
import streamlit as st
from huggingface_hub import login
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb
import nest_asyncio
import torch
import re

# Apply nest_asyncio to prevent event loop issues
nest_asyncio.apply()

# Hugging Face login (replace with your token retrieval method in a non-Kaggle environment)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")
login(token="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")

# Streamlit interface begins here
st.title("PDF Question Answering with AI")

# Allow users to upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Load the model
model_id = 'meta-llama/Llama-3.1-8B-Instruct'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

# Initialize ChromaDB client and SentenceTransformer
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v2')
client = chromadb.Client()
pdf_collection = client.create_collection("pdf_documents")

# Function to process PDFs and split text
def process_pdfs(uploaded_files):
    pdf_data = []
    for file in uploaded_files:
        data_loader = PyPDFLoader(file)
        data = data_loader.load()
        pdf_data.append(data)
    
    # Split documents
    page_contents = [doc for i in range(len(pdf_data)) for doc in pdf_data[i]]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    documents = text_splitter.split_documents(page_contents)
    return documents

# If files are uploaded, process them
if uploaded_files:
    st.write("Processing uploaded PDFs...")
    documents = process_pdfs(uploaded_files)

    # Embed documents and add to ChromaDB
    pdf_embedding = embedder.encode([doc.page_content for doc in documents])
    ids = [str(i) for i in range(len(documents))]
    pdf_collection.add(
        ids=ids,
        documents=[doc.page_content for doc in documents],
        embeddings=pdf_embedding
    )
    st.write(f"Processed {len(documents)} document segments.")

# Function to query ChromaDB
def query_chroma(query):
    query_embedding = embedder.encode([query])
    results = pdf_collection.query(query_embeddings=query_embedding, n_results=10)
    return results['documents']

# User query input
query = st.text_input("Ask a question based on the uploaded PDFs:")

# Process the query when entered
if query:
    retrieved_docs = query_chroma(query)
    retrieved_context = " ".join([doc for doc in retrieved_docs])
    
    # Format input for Llama model
    formatted_input = f"""
    Context:
    {retrieved_context}

    Instructions:
    "You are a helpful assistant that extracts the most useful information from the context above. Also bring in extra relevant information to the user query from outside the given context if necessary."
    "Ignore any references or links in the context."
    "Answer with the same language as the question."

    Question:
    {query}

    Answer:
    """

    # Tokenize and generate output from the model
    inputs = tokenizer(formatted_input, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        no_repeat_ngram_size=4,
        repetition_penalty=1.2,
        temperature=0.0,
        do_sample=False
    )

    # Decode and clean the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer_start = response.find("Answer:") + len("Answer: ")
    cleaned_answer = response[answer_start:].strip()

    # Display the answer
    st.write("### Answer:")
    st.write(cleaned_answer)