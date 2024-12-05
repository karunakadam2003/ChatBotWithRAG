from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup
import numpy as np

# Step 1: Initialize the Embedding Model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

# Step 2: Generate Cluster Centroids Dynamically
def generate_clusters(data, model, n_clusters=2):
    embeddings = model.encode(data)  # Generate embeddings for the dataset
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_  # Return the centroids

# Step 3: Embed Query
def embed_query(query, model):
    return model.encode([query])[0]

# Step 4: Find the Closest Cluster
def find_closest_cluster(query_vector, cluster_centroids):
    similarities = cosine_similarity([query_vector], cluster_centroids)
    closest_cluster_index = np.argmax(similarities)
    return closest_cluster_index

# Step 5: Fetch URLs for the Closest Cluster
def fetch_urls(cluster_index, url_database):
    return url_database.get(cluster_index, [])

# Step 6: Fetch Data from URLs
def fetch_data_from_urls(urls):
    fetched_data = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                fetched_data.append(soup.get_text())  # Extract text from HTML
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
    return fetched_data

# URL Database for Clusters
url_database = {
    0: [
        "https://www.wikipedia.org",       # Wikipedia main page
        "https://www.python.org",          # Python official website
    ],
    1: [
        "https://www.kaggle.com",          # Kaggle for data science projects
        "https://www.stackoverflow.com",  # Stack Overflow for programming Q&A
    ],
}

# Mock Dataset for Clustering
data = [
    "Python programming resources",
    "Learn data science with Kaggle",
    "Stack Overflow for developers",
    "Wikipedia for general knowledge",
]

# Main Logic
if __name__ == "__main__":
    # Step 1: Generate Cluster Centroids
    cluster_centroids = generate_clusters(data, embedding_model, n_clusters=2)

    # Step 2: Input Query
    query = "How to implement clustering in Python?"

    # Step 3: Convert Query to Embedding
    query_vector = embed_query(query, embedding_model)

    # Step 4: Find Closest Cluster
    closest_cluster = find_closest_cluster(query_vector, cluster_centroids)

    # Step 5: Retrieve URLs from the Closest Cluster
    urls = fetch_urls(closest_cluster, url_database)

    # Step 6: Fetch Data from URLs
    fetched_data = fetch_data_from_urls(urls)

    # Output Results
    print(f"Closest Cluster: {closest_cluster}")
    print("Fetched Data:")
    for data in fetched_data:
        print(data[:200])  # Print the first 200 characters of each result
