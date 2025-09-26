from jax import numpy as jnp
from utils import clean_text




# Read corpus (UTF-8) and split into lines
with open("tinymoliere.txt", "r", encoding="utf-8") as f:
    corpus = [line for line in f.read().splitlines() if line.strip()]

# Train/valid split (line level)
split_idx = int(0.9 * len(corpus))
train_raw = corpus[:split_idx]
valid_raw = corpus[split_idx:]

# Clean and tokenize lines into lists of words
train_corpus = [clean_text(line).split() for line in train_raw]
valid_corpus = [clean_text(line).split() for line in valid_raw]

# Count word occurrences on training data
all_words = [w for sent in train_corpus for w in sent]
word_counts = {}
for w in all_words:
    word_counts[w] = word_counts.get(w, 0) + 1

# Build vocabulary from training words
vocab = sorted(word_counts.keys())
V = len(vocab)
total_words = len(all_words)

# Index mappings
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}

# Count vector aligned with vocab order
count_vector = jnp.zeros(V, dtype=jnp.int32)
for word, cnt in word_counts.items():
    idx = word_to_id[word]
    count_vector = count_vector.at[idx].set(cnt)

# Without smoothing
prob_unsmoothed = count_vector.astype(jnp.float32) / float(total_words)


# Perplexity on validation
log_prob = 0.0
total_tokens = 0
for sentence in valid_corpus:
    for word in sentence:
        if word in word_to_id:
            p = float(prob_unsmoothed[word_to_id[word]])
            log_prob += jnp.log(p)
            total_tokens += 1
        else:
            log_prob += -float("inf") # inf penalty for OOV
            total_tokens += 1


perp_unsmoothed = float(jnp.exp(-log_prob / total_tokens)) if total_tokens > 0 else float("inf")
print(f"Validation Perplexity without smoothing: {perp_unsmoothed:.4f}")

# With Laplace (add-1) smoothing
prob_laplace = (count_vector.astype(jnp.float32) + 1.0) / float(total_words + V)

# Perplexity with Laplace
log_prob = 0.0
total_tokens = 0
for sentence in valid_corpus:
    for word in sentence:
        total_tokens += 1
        if word in word_to_id:
            p = float(prob_laplace[word_to_id[word]])
            log_prob += jnp.log(p)
        else:
            p = float(1 / (jnp.sum(prob_laplace) + 1e-9))
            log_prob += jnp.log(p)

perp_laplace = float(jnp.exp(-log_prob / total_tokens)) if total_tokens > 0 else float("inf")
print(f"Validation Perplexity with Laplace smoothing: {perp_laplace:.4f}")

# Top-N display (change N as needed)
N = 20

records = []
for word, p1, p2 in zip(vocab, prob_unsmoothed, prob_laplace):
    diff = float(p2 - p1)
    records.append((word, float(p1), float(p2), diff))

# Sort criteria: by unsmoothed probability descending
top_by_unsmoothed = sorted(records, key=lambda r: r[1], reverse=True)[:N]

print(f"{'Mot':<12} | {'Sans lissage':<14} | {'Avec lissage':<14} | {'Diff (lissé - brut)':<18}")
print("-" * 70)
for word, p1, p2, diff in top_by_unsmoothed:
    print(f"{word:<12} | {p1:<14.6f} | {p2:<14.6f} | {diff:+.6f}")