import numpy as np
import time
import fitz  
import nltk
from nltk.tokenize import sent_tokenize
from vector_store import VectorStore
import faiss

nltk.download('punkt')
nltk.download('punkt_tab')

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + " "  
    return text.strip()

# Process PDF
pdf_path = "cough.pdf" 
text = extract_text_from_pdf(pdf_path)
sentences = sent_tokenize(text)  

# Initialize Vector Store
vector_store = VectorStore()

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

# Store vectors in VectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Search Similarity using VectorStore
query_sentence = "I love my Mother"
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

# Normalize vectors to unit length (important for cosine similarity)
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

vectors_array = normalize_vectors(vectors_array)
query_vector = normalize_vectors(np.array([query_vector])) 

index = faiss.IndexFlatIP(len(vocabulary))  # Inner Product for cosine similarity
index.add(vectors_array)

start_time = time.time()
k = 2
distances, indices = index.search(query_vector, k)
end_time = time.time()

print("\nSimilar Sentences using FAISS:")
for i in range(k):
    sentence = list(sentence_vectors.keys())[indices[0][i]]
    similarity = distances[0][i] 
    print(f"{sentence}: Similarity = {similarity:.4f}")
print(f"Execution Time for FAISS: {end_time - start_time:.4f} seconds")
