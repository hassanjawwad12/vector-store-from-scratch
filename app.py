import numpy as np
import time
from vector_store import VectorStore
import faiss

# Create a VectorStore instance
vector_store = VectorStore()

# Define your sentences
sentences = [
    "I eat mango",
    "mango is my favorite fruit",
    "mango, apple, oranges are fruits",
    "fruits are good for health"
]

# Tokenization and Vocabulary Creation
vocabulary = set()

for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}

for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# Storing in VectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for Similarity using VectorStore
query_sentence = "Mango is the best fruit"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1

start_time = time.time()
similar_sentences_custom = vector_store.get_similar_vectors(query_vector, num_results=2)
end_time = time.time()

print("Query Sentence:", query_sentence)
print("Similar Sentences using Custom Store:")
for sentence, similarity in similar_sentences_custom:
    print(f"{sentence}: Similarity = {similarity:.4f}")
print(f"Execution Time for Custom Store: {end_time - start_time:.4f} seconds")

# Using FAISS
vectors = list(sentence_vectors.values())
vectors_array = np.array(vectors).astype('float32')

index = faiss.IndexFlatL2(len(vocabulary))
index.add(vectors_array)

start_time = time.time()
k = 2
distances, indices = index.search(np.array([query_vector]), k)
end_time = time.time()

print("\nSimilar Sentences using FAISS:")
for i in range(k):
    sentence = list(sentence_vectors.keys())[indices[0][i]]
    similarity = 1 - distances[0][i] / (len(vocabulary) ** 2)  # Normalize similarity
    print(f"{sentence}: Similarity = {similarity:.4f}")
print(f"Execution Time for FAISS: {end_time - start_time:.4f} seconds")