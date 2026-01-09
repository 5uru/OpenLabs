import numpy as np
import PIL as pil
from sentence_transformers import SentenceTransformer
from uform import Modality, get_model
from usearch.index import Index

processors, models = get_model("unum-cloud/uform3-image-text-english-small")


# 1. Initialize the Embedding Model
# This converts text to vectors (embeddings)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Your Original Data (The Text)
documents = [
    "USearch is a high-performance vector search engine.",
    "Python is a popular programming language.",
    "The quick brown fox jumps over the lazy dog.",
    "Vector databases are essential for AI applications.",
]

# 3. Create a Lookup Table (Map)
# This is crucial! We map a unique ID (integer) to the text.
# In a real app, this would likely be a Database ID (SQL/NoSQL).
id_to_text = {i: text for i, text in enumerate(documents)}

print(f"Lookup Table: {id_to_text}")
# Output: {0: 'USearch is...', 1: 'Python is...', ...}


# 4. Generate Embeddings
vectors = model.encode(documents)

# 5. Initialize and Populate USearch Index
# We use the same keys (0, 1, 2, 3) as our lookup table
index = Index(ndim=384, metric="cos", dtype="f16")
keys = np.array(list(id_to_text.keys()))
index.add(keys, vectors)


# 6. Perform a Search
query_text = "I need a fast search engine"
query_vector = model.encode([query_text])[0]

# Search for the top 2 most similar vectors
matches = index.search(query_vector, 2)

print(f"\n--- Search Results for: '{query_text}' ---")

# 7. Retrieve the Original Text
for match in matches:
    # match.key is the ID we stored earlier
    doc_id = match.key
    distance = match.distance

    # USE THE ID TO GET THE TEXT FROM OUR LOOKUP TABLE
    original_text = id_to_text[doc_id]

    print(f"ID: {doc_id}")
    print(f"Text: {original_text}")
    print(f"Distance: {distance:.4f}")
    print("-" * 30)
