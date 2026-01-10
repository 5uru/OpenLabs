from datasets import load_dataset
from vector_db import VectorDB

ds = load_dataset("aimped/medical-translation-test-set")["en_fr"]


# Shuffle the dataset
ds = ds.shuffle(seed=42)
# split into train_set, val_set, test_set
train_size = int(0.8 * len(ds))
val_size = int(0.1 * len(ds))
test_size = len(ds) - train_size - val_size

train_set = ds.select(range(train_size))
val_set = ds.select(range(train_size, train_size + val_size))
test_set = ds.select(range(train_size + val_size, len(ds)))

print(f"Train set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

vector_db = VectorDB()
for item in train_set:
    text = f"English: {item['source']} French: {item['target']}"
    vector_db.add_entry(text, item["target"])
print("Finished adding entries to the vector database.")

for item in test_set:
    query = f"Translate to French: {item['source']}"
    results = vector_db.get_entry(query, top_k=5)
    print(f"Query: {query}")
    print("Number of results:", len(results))
    print()
