# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import chainlit as cl

# import pandas as pd

# from transformers import AutoTokenizer, T5ForConditionalGeneration

# # Corrected model and tokenizer loading for T5-compatible configurations
# model_name = "google/flan-t5-small"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)


# # Load a pre-trained model for text generation (you can choose GPT-2, Bloom, etc.)
# # model_name = "bigscience/bloom-560m"
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Load a SentenceTransformer model for embedding (for RAG)
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Load the financial data with pipe delimiter
# data = pd.read_csv("financial_data.txt", delimiter="|")

# # Create embeddings for your financial text data (for retrieval)
# financial_embeddings = embedding_model.encode(data['Financial Summary'].tolist(), convert_to_tensor=False)

# # Initialize FAISS index for fast similarity search
# embedding_dim = financial_embeddings.shape[1]  # Dimensionality of the embeddings
# faiss_index = faiss.IndexFlatL2(embedding_dim)  # Use L2 distance for similarity
# faiss_index.add(np.array(financial_embeddings))


# def retrieve_relevant_data(query, top_k=3):
#     # Embed the query
#     query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    
#     # Retrieve the top_k most similar financial data points
#     distances, indices = faiss_index.search(np.array(query_embedding), top_k)
    
#     # Fetch the relevant financial texts
#     relevant_texts = data.iloc[indices[0]]['Financial Summary'].tolist()
    
#     # Concatenate the relevant data
#     context = "\n".join(relevant_texts)
    
#     return context

# def generate_response(query):
#     # Retrieve relevant financial data
#     relevant_data = retrieve_relevant_data(query)
    
#     # Combine the query and the relevant context
#     input_text = f"{relevant_data}\n\nUser Question: {query}"
    
#     # Generate a response using the language model
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
#     outputs = model.generate(**inputs, max_length=150)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return response



# @cl.on_message
# async def main(query: str):
#     # Call the function to generate a response based on RAG
#     response = generate_response(query)
    
#     # Send the response back to the Chainlit interface
#     await cl.Message(content=response).send()
# # Run with this: python -m chainlit run app.py



#==================================== Without RAG ===========================================

import os
import chainlit as cl
from transformers import pipeline
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the text generation model
# generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)
# generation_model = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
generation_model = pipeline("text2text-generation", model="google/flan-t5-xxl", device=-1)

# Load financial data from the file
def load_documents(file_path):
    data = pd.read_csv(file_path, delimiter="|")
    documents = data['Financial Summary'].tolist()
    return documents

# Load documents from 'financial_data.txt'
documents = load_documents("financial_data.txt")

# Fetch data from a web source (placeholder function for illustration)
def fetch_web_data(query):
    api_url = "https://api.example.com/search"
    params = {"q": query, "num_results": 3}
    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    try:
        response = session.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json()
        web_content = [result['snippet'] for result in results.get("items", [])]
        return "\n".join(web_content)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching web data: {e}")
        return ""

# @cl.on_message decorator to handle incoming messages (queries)
@cl.on_message
async def on_message(message):
    try:
        user_query = message.content
    except AttributeError:
        logging.error("The message object does not have a 'content' attribute")
        return

    logging.info(f"User query: {user_query}")

    # Generate response based on the user's query directly
    prompt = f"Answer the following question based on your knowledge.\n\nQuestion: {user_query}"
    response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

    # Send the generated response back to the user
    response_message = f"**Response:** {response}\n\n"
    await cl.Message(content=response_message).send()
