# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 07:33:47 2024

@author: akash-
"""

import nltk  # For text processing
import numpy as np  # For numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF embeddings (simple example)
from sklearn.metrics.pairwise import cosine_similarity  # For similarity search

# 1. Data Loading (Example - using a few sentences as a dataset)
dataset = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
    "Akash is the son of Mukesh ambani"
]

# 2. Chunking (In this simple example, each sentence is a chunk)
chunks = dataset  # No further splitting needed for this example

# 3. Embedding Creation (Using TF-IDF - a simple method for demonstration)
vectorizer = TfidfVectorizer()
chunk_embeddings = vectorizer.fit_transform(chunks)

# 4. Store Embeddings (In this example, storing in a simple list - a real application would use a vector database)
vector_database = chunk_embeddings.toarray().tolist() # Convert sparse matrix to list

# 5. Retrieval (Using cosine similarity for demonstration)
def retrieve_relevant_chunks(query, k=2):  # Retrieve top k chunks
    query_embedding = vectorizer.transform([query]).toarray()[0]
    similarities = cosine_similarity([query_embedding], vector_database)
    top_indices = np.argsort(similarities[0])[::-1][:k]  # Get indices of top k similar chunks
    return [chunks[i] for i in top_indices]

# Example Usage:
query = "who is akash"
relevant_chunks = retrieve_relevant_chunks(query)
print(f"Query: {query}")
print("Relevant Chunks:")
for chunk in relevant_chunks:
    print(chunk)


# --- Improvements for a Real-World Application ---

# Embedding Model:
# - Use more advanced embedding models like Sentence Transformers (sentence-transformers)
#   for better semantic representation.

# Vector Database:
# - Use a dedicated vector database like FAISS, Pinecone, Chroma, or Weaviate for efficient storage and retrieval.

# Chunking:
# - Implement more sophisticated chunking strategies (e.g., by sentence, paragraph, or sliding window)
#   depending on the dataset and the context window of the embedding model.

# Retrieval:
# - Implement more advanced retrieval methods (e.g., Maximum Marginal Relevance) to diversify the retrieved chunks.

# Evaluation:
# - Add evaluation metrics to assess the performance of the retrieval system.