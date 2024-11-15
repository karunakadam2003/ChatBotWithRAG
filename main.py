# import os
# import chainlit as cl
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import chromadb
# from chromadb.config import Settings

# # Initialize Chroma DB client
# chroma_client = chromadb.Client(Settings())

# # Create or retrieve a collection in Chroma DB
# collection_name = "document_collection"
# collection = chroma_client.create_collection(name=collection_name)

# # Initialize embedding and generation models
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)

# # Load documents from the file and add them to Chroma DB
# def load_documents(file_path):
#     with open(file_path, "r") as file:
#         documents = [line.strip() for line in file if line.strip()]

#     # Embed documents and add them to Chroma DB collection
#     embeddings = embedding_model.encode(documents).tolist()  # Convert to list format for Chroma DB

#     # Generate unique IDs for each document (using index `i` as a string)
#     ids = [str(i) for i in range(len(documents))]

#     # Add documents to Chroma DB with embeddings and metadata
#     for i, doc in enumerate(documents):
#         collection.add(
#             documents=[doc], 
#             metadatas=[{"id": ids[i]}],  # Store metadata with the generated ID
#             embeddings=[embeddings[i]],   # Corresponding embedding for the document
#             ids=[ids[i]]                 # Pass the IDs for each document
#         )

# # Load and index documents from 'financial_data.txt'
# load_documents("financial_data.txt")

# #@cl.on_message
# def on_message(message):
#     # Debugging: Check the structure of the message object
#     print(f"Received message object: {message}")

#     # Ensure that we access the correct text content from the message
#     try:
#         # Extract the text content from the message
#         user_query = message.get("text")  # Accessing the 'text' field directly
#     except AttributeError:
#         print("The message object does not have a 'text' attribute")
#         return

#     print(f"User query: {user_query}")

#     # Step 1: Embed the user's query
#     user_query_embedding = embedding_model.encode([user_query]).tolist()

#     # Step 2: Retrieve the top 3 relevant documents from Chroma DB
#     results = collection.query(
#         query_embeddings=user_query_embedding,
#         n_results=3  # Top 3 relevant documents
#     )
#     relevant_docs = [result["document"] for result in results["documents"][0]]

#     # Step 3: Use relevant documents as context for response generation
#     context = "\n".join(relevant_docs)
#     prompt = f"Answer the question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {user_query}"
#     response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

#     # Step 4: Send the generated response back to the user
#     cl.send_message(response)

#     # Step 5: Show relevant document titles or placeholders
#     for doc in relevant_docs:
#         cl.send_message(f"Relevant document: {doc}")

#8888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888

# from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import pandas as pd
# import chainlit as cl

# # Set up models
# model_name = "t5-small"  # Choose a compatible model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # Load embedding model for RAG
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# # Load financial data
# data = pd.read_csv("financial_data.txt", delimiter="|")
# financial_texts = data['Financial Summary'].tolist()

# # Create embeddings for financial data
# financial_embeddings = embedding_model.encode(financial_texts, convert_to_tensor=False)

# # Initialize FAISS index
# embedding_dim = financial_embeddings.shape[1]
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# faiss_index.add(np.array(financial_embeddings))

# def retrieve_relevant_data(query, top_k=3):
#     print("Query Type:", type(query))  # Debugging statement
#     query_text = query.content if hasattr(query, 'content') else query  # Ensure we get the string content

#     # Embed the user query
#     query_embedding = embedding_model.encode([query_text], convert_to_tensor=False)

#     # Search for top_k most relevant documents
#     distances, indices = faiss_index.search(np.array(query_embedding), top_k)
#     relevant_texts = [financial_texts[i] for i in indices[0]]
    
#     # Combine relevant texts
#     context = "\n".join(relevant_texts)
#     return context

# def generate_response(query):
#     # Retrieve relevant context
#     relevant_data = retrieve_relevant_data(query)
    
#     # Combine query and context for model input
#     input_text = f"Relevant Financial Data:\n{relevant_data}\n\nUser Question: {query}"
    
#     # Generate response
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
#     outputs = model.generate(**inputs, max_length=150)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return response

# @cl.on_message
# async def main(query):
#     response = generate_response(query)
#     await cl.Message(content=response).send()



#88888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888888
# import os
# import chainlit as cl
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import chromadb
# from chromadb.config import Settings
# import pandas as pd

# # Initialize Chroma DB client
# chroma_client = chromadb.Client(Settings())

# # Create or retrieve a collection in Chroma DB
# collection_name = "document_collection"
# collection = chroma_client.create_collection(name=collection_name)

# # Initialize embedding and generation models
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)

# # Load financial data from the file
# def load_documents(file_path):
#     # Read the financial data into a DataFrame
#     data = pd.read_csv(file_path, delimiter="|")
    
#     # Extract financial summaries (assuming the column 'Financial Summary' contains relevant text)
#     documents = data['Financial Summary'].tolist()

#     # Embed documents using the SentenceTransformer model
#     embeddings = embedding_model.encode(documents).tolist()  # Convert to list format for Chroma DB

#     # Generate unique IDs for each document
#     ids = [str(i) for i in range(len(documents))]

#     # Add documents to Chroma DB with embeddings and metadata
#     for i, doc in enumerate(documents):
#         collection.add(
#             documents=[doc], 
#             metadatas=[{"id": ids[i]}],  # Store metadata with the generated ID
#             embeddings=[embeddings[i]],   # Corresponding embedding for the document
#             ids=[ids[i]]                 # Pass the IDs for each document
#         )

# # Load and index documents from 'financial_data.txt'
# load_documents("financial_data.txt")

# # @cl.on_message decorator to handle incoming messages (queries)
# @cl.on_message
# async def on_message(message):
#     # Debugging: Check the structure of the message object
#     print(f"Received message object: {message}")

#     # Ensure we access the correct text content from the message
#     try:
#         user_query = message.get("text")  # Accessing the 'text' field directly
#     except AttributeError:
#         print("The message object does not have a 'text' attribute")
#         return

#     print(f"User query: {user_query}")

#     # Step 1: Embed the user's query
#     user_query_embedding = embedding_model.encode([user_query]).tolist()

#     # Step 2: Retrieve the top 3 relevant documents from Chroma DB
#     results = collection.query(
#         query_embeddings=user_query_embedding,
#         n_results=3  # Top 3 relevant documents
#     )
    
#     # Get the relevant documents from the results
#     relevant_docs = [result["document"] for result in results["documents"][0]]

#     # Step 3: Use relevant documents as context for response generation
#     context = "\n".join(relevant_docs)
#     prompt = f"Answer the question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {user_query}"

#     # Generate response based on the context and query
#     response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

#     # Step 4: Send the generated response back to the user
#     await cl.Message(content=response).send()

#     # Step 5: Show relevant document titles or placeholders (optional)
#     for doc in relevant_docs:
#         await cl.Message(content=f"Relevant document: {doc}").send()



# import os
# import chainlit as cl
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import chromadb
# from chromadb.config import Settings
# import pandas as pd

# # Initialize Chroma DB client
# chroma_client = chromadb.Client(Settings())

# # Create or retrieve a collection in Chroma DB
# collection_name = "document_collection"
# collection = chroma_client.create_collection(name=collection_name)

# # Initialize embedding and generation models
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)

# # Load financial data from the file
# def load_documents(file_path):
#     # Read the financial data into a DataFrame
#     data = pd.read_csv(file_path, delimiter="|")
    
#     # Extract financial summaries (assuming the column 'Financial Summary' contains relevant text)
#     documents = data['Financial Summary'].tolist()

#     # Embed documents using the SentenceTransformer model
#     embeddings = embedding_model.encode(documents).tolist()  # Convert to list format for Chroma DB

#     # Generate unique IDs for each document
#     ids = [str(i) for i in range(len(documents))]

#     # Add documents to Chroma DB with embeddings and metadata
#     for i, doc in enumerate(documents):
#         collection.add(
#             documents=[doc], 
#             metadatas=[{"id": ids[i]}],  # Store metadata with the generated ID
#             embeddings=[embeddings[i]],   # Corresponding embedding for the document
#             ids=[ids[i]]                 # Pass the IDs for each document
#         )

# # Load and index documents from 'financial_data.txt'
# load_documents("financial_data.txt")

# # @cl.on_message decorator to handle incoming messages (queries)
# @cl.on_message
# async def on_message(message):
#     # Debugging: Check the structure of the message object
#     print(f"Received message object: {message}")

#     # Ensure we access the correct text content from the message
#     try:
#         user_query = message.get("text")  # Accessing the 'text' field directly
#     except AttributeError:
#         print("The message object does not have a 'text' attribute")
#         return

#     print(f"User query: {user_query}")

#     # Step 1: Embed the user's query
#     user_query_embedding = embedding_model.encode([user_query]).tolist()

#     # Step 2: Retrieve the top 3 relevant documents from Chroma DB
#     results = collection.query(
#         query_embeddings=user_query_embedding,
#         n_results=3  # Top 3 relevant documents
#     )
    
#     # Get the relevant documents from the results
#     relevant_docs = [result["document"] for result in results["documents"][0]]

#     # Step 3: Use relevant documents as context for response generation
#     context = "\n".join(relevant_docs)
#     prompt = f"Answer the question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {user_query}"

#     # Generate response based on the context and query
#     response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

#     # Step 4: Send the generated response and relevant documents back to the user
#     response_message = f"**Response:** {response}\n\n**Top 3 Relevant Documents:**\n"
#     for doc in relevant_docs:
#         response_message += f"- {doc}\n"
    
#     await cl.Message(content=response_message).send()




# import os
# import chainlit as cl
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# import chromadb
# from chromadb.config import Settings
# import pandas as pd
# import requests  # For web data fetching (if using an API)

# # Initialize Chroma DB client
# chroma_client = chromadb.Client(Settings())
# collection_name = "document_collection"
# collection = chroma_client.create_collection(name=collection_name)

# # Initialize embedding and generation models
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)

# # Load financial data from the file
# def load_documents(file_path):
#     data = pd.read_csv(file_path, delimiter="|")
#     documents = data['Financial Summary'].tolist()
#     embeddings = embedding_model.encode(documents).tolist()
#     ids = [str(i) for i in range(len(documents))]

#     for i, doc in enumerate(documents):
#         collection.add(
#             documents=[doc], 
#             metadatas=[{"id": ids[i]}],
#             embeddings=[embeddings[i]],
#             ids=[ids[i]]
#         )

# # Load and index documents from 'financial_data.txt'
# load_documents("financial_data.txt")

# # Fetch data from a web source (placeholder function for illustration)
# def fetch_web_data(query):
#     # Example API endpoint
#     api_url = "https://api.example.com/search"
#     params = {"q": query, "num_results": 3}  # Customize as per the API requirements
#     response = requests.get(api_url, params=params)

#     # Check response
#     if response.status_code == 200:
#         results = response.json()  # Assuming the API returns JSON data
#         web_content = [result['snippet'] for result in results["items"]]
#         return "\n".join(web_content)
#     else:
#         print("Error fetching web data:", response.status_code)
#         return ""

# # @cl.on_message decorator to handle incoming messages (queries)
# @cl.on_message
# async def on_message(message):
#     # Access the content of the message
#     try:
#         user_query = message.content  # Accessing the 'content' attribute directly
#     except AttributeError:
#         print("The message object does not have a 'content' attribute")
#         return

#     print(f"User query: {user_query}")

#     # Step 1: Embed the user's query
#     user_query_embedding = embedding_model.encode([user_query]).tolist()

#     # Step 2: Retrieve the top 3 relevant documents from Chroma DB
#     results = collection.query(
#         query_embeddings=user_query_embedding,
#         n_results=3  # Top 3 relevant documents
#     )
    
#     # Debugging: Print the structure of results to understand it
#     print("Query Results:", results)

#     # Extract relevant documents from results
#     relevant_docs = results['documents'][0] if 'documents' in results and results['documents'] else []

#      # If no relevant documents are found, fetch data from the web
#     if not relevant_docs:
#         print("No relevant documents found in local data. Fetching from web...")
#         context = fetch_web_data(user_query)
#     else:
#         context = "\n".join(relevant_docs)

#     # Step 3: Use relevant documents or web data as context for response generation
#     prompt = f"Answer the question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {user_query}"

#     # Generate response based on the context and query
#     response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

#     # Step 4: Send the generated response and relevant documents (or web content) back to the user
#     response_message = f"**Response:** {response}\n\n"
#     if relevant_docs:
#         response_message += "**Top 3 Relevant Documents:**\n"
#         for doc in relevant_docs:
#             response_message += f"- {doc}\n"
#     else:
#         response_message += "**Web Data:**\n" + context

#     await cl.Message(content=response_message).send()


#==================================== Final Version ===========================================
import os
import chainlit as cl
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
import logging
# Fetch data from a web source (placeholder function for illustration)
from requests_html import HTMLSession
from bs4 import BeautifulSoup

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Chroma DB client
chroma_client = chromadb.Client(Settings())
collection_name = "document_collection"
collection = chroma_client.create_collection(name=collection_name)

# Initialize embedding and generation models
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
generation_model = pipeline("text-generation", model="google/flan-t5-large", device=-1)

# Load financial data from the file
def load_documents(file_path):
    data = pd.read_csv(file_path, delimiter="|")
    documents = data['Financial Summary'].tolist()
    embeddings = embedding_model.encode(documents).tolist()
    ids = [str(i) for i in range(len(documents))]

    for i, doc in enumerate(documents):
        collection.add(
            documents=[doc],
            metadatas=[{"id": ids[i]}],
            embeddings=[embeddings[i]],
            ids=[ids[i]]
        )

# Load and index documents from 'financial_data.txt'
load_documents("financial_data.txt")


from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
import logging

# Fetch data from the web using Beautiful Soup asynchronously
async def fetch_web_data(query):
    search_url = f"https://www.google.com/search?q={query}"
    search_url = "https://storage.googleapis.com/chromium-browser-snapshots/Win_x64/<valid_version>/chrome-win.zip"
    # Initialize an asynchronous session
    session = AsyncHTMLSession()
    try:
        # Send the search request and render JavaScript asynchronously
        response = await session.get(search_url)
        await response.html.arender(timeout=20)  # Render JS content, adjust timeout as needed
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.html.html, "html.parser")
        
        # Select relevant text snippets from search results
        snippets = []
        for result in soup.select("div.BNeawe.s3v9rd.AP7Wnd"):  # Adjust selector for the content you need
            snippets.append(result.text)
            if len(snippets) >= 3:  # Limit to top 3 snippets
                break
        
        # Join snippets for context
        return "\n".join(snippets)
    except Exception as e:
        logging.error(f"Error fetching web data with Beautiful Soup: {e}")
        return ""
    finally:
        await session.close()  # Close the asynchronous session

# @cl.on_message decorator to handle incoming messages (queries)
@cl.on_message
async def on_message(message):
    try:
        user_query = message.content
    except AttributeError:
        logging.error("The message object does not have a 'content' attribute")
        return

    logging.info(f"User query: {user_query}")

    # Embed the user's query
    user_query_embedding = embedding_model.encode([user_query]).tolist()

    # Retrieve the top 3 relevant documents from Chroma DB
    relevance_threshold = 0.8  # Adjust the threshold as needed
    results = collection.query(query_embeddings=user_query_embedding, n_results=3)

    # Log the entire results object to inspect its structure
    logging.info(f"Query Results: {results}")

    # Check if 'documents' and 'distances' keys exist in the results
    if 'documents' in results and 'distances' in results:
        logging.info("Retrieved Documents and Scores:")
        
        # Pair the documents with their corresponding scores
        paired_results = list(zip(results['documents'][0], results['distances'][0]))
        
        relevant_docs = []
        for doc, score in paired_results:
            logging.info(f"Document: {doc}, Score: {score}")
            if score >= relevance_threshold:
                relevant_docs.append((doc, score))
        
        # Log relevant documents
        if relevant_docs:
            logging.info("Relevant Documents:")
            for doc, score in relevant_docs:
                logging.info(f"- {doc} (Score: {score})")
        else:
            logging.info("No relevant documents found with score above the threshold.")
    else:
        logging.warning("No 'documents' or 'distances' keys found in results.")

    # If no relevant documents are found, fetch data from the web
    # if not relevant_docs:
    #     logging.info("No relevant documents found in local data. Fetching from web...")
    #     context = fetch_web_data(user_query)
    # else:
    #     context = "\n".join([doc for doc, _ in relevant_docs])

    # If no relevant documents found, fetch data from the web
    # if not relevant_docs:
    #     logging.info("No relevant documents found in local data. Fetching from web...")
    #     context = await fetch_web_data(user_query)  # Await the async fetch_web_data call
    # else:
    #     context = "\n".join([doc for doc, _ in relevant_docs])

    logging.info("No relevant documents found in local data. Fetching from web...")
    context = await fetch_web_data(user_query)  # Await the async fetch_web_data call
    context += "\n".join([doc for doc, _ in relevant_docs])


    # Generate response based on context and query
    prompt = f"Answer the question based on the context provided.\n\nContext:\n{context}\n\nQuestion: {user_query}"
    response = generation_model(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]

    # Send the generated response and relevant documents (or web content) back to the user
    response_message = f"**Response:** {response}\n\n"
    print(response_message)
    # if relevant_docs:
    #     response_message += f"**Top Relevant Documents (Score >= {relevance_threshold}):**\n"
    #     for doc, score in relevant_docs:
    #         response_message += f"- {doc} (Score: {score})\n"
    # else:
    #     response_message += "**Web Data:**\n" + context
    response_message += f"**Top Relevant Documents (Score >= {relevance_threshold}):**\n"
    for doc, score in relevant_docs:
        response_message += f"- {doc} (Score: {score})\n"
    response_message += "**Web Data:**\n" + context

    await cl.Message(content=response_message).send()

