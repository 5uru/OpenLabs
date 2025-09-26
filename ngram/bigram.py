from pathlib import Path
from itertools import chain
from collections import Counter
from jax import numpy as jnp
from utils import clean_text
import jax.random as jr

corpus_path = "tinymoliere.txt"
lines = [ln.strip() for ln in Path(corpus_path).read_text(encoding="utf-8").splitlines() if ln.strip()]

split_idx = int(0.9 * len(lines))
train_raw = lines[:split_idx]
valid_raw = lines[split_idx:]

train_tokens = [clean_text(l).split() for l in train_raw]
valid_tokens = [clean_text(l).split() for l in valid_raw]

# Build UNK handling
min_freq = 30         # collapse words with freq < min_freq
freq = Counter(chain.from_iterable(train_tokens))

# Replace rare words with <unk> in training
train_norm = [[(w if freq[w] >= min_freq else "<unk>") for w in sent] for sent in train_tokens]

# Add sentence boundaries
train_proc = [["<s>"] + sent + ["</s>"] for sent in train_norm]

# Vocab: include <unk>, <s>, </s>
vocab = sorted(set(chain.from_iterable(train_proc)))
if "<unk>" not in vocab:
    vocab.append("<unk>")
vocab = sorted(vocab)
V = len(vocab)

word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}

# Collect bigram counts
w1_ids, w2_ids = [], []
for sent in train_proc:
    ids = [word_to_id[w] for w in sent]
    w1_ids.extend(ids[:-1])
    w2_ids.extend(ids[1:])

if w1_ids:
    w1 = jnp.array(w1_ids, dtype=jnp.int32)
    w2 = jnp.array(w2_ids, dtype=jnp.int32)
    flat = w1 * V + w2
    counts_flat = jnp.bincount(flat, length=V * V)
    count_matrix = counts_flat.reshape(V, V)
else:
    count_matrix = jnp.zeros((V, V), dtype=jnp.int32)

# Add-alpha smoothing (Laplace when alpha=1.0)
alpha = 1.0
row_sums = count_matrix.sum(axis=1, keepdims=True)
prob_smoothed = (count_matrix.astype(jnp.float32) + alpha) / (row_sums + alpha * V)

# Validation perplexity using UNK mapping
def map_token(w: str) -> int:
    if w in word_to_id:
        return word_to_id[w]
    return word_to_id["<unk>"]

log_prob = 0.0
total = 0
for sent in valid_tokens:
    sent_mapped = ["<s>"] + [ (w if freq.get(w,0) >= min_freq else "<unk>") for w in sent ] + ["</s>"]
    for i in range(1, len(sent_mapped)):
        i1 = map_token(sent_mapped[i-1])
        i2 = map_token(sent_mapped[i])
        p = prob_smoothed[i1, i2]
        log_prob += jnp.log(p)
        total += 1

perplexity = float(jnp.exp(-log_prob / total)) if total > 0 else float("inf")
print(f"Validation Perplexity (add-alpha α={alpha}, UNK): {perplexity:.4f}")

# --- Top-N most frequent bigram pairs excluding special tokens ---
N = 20

special_ids = {
        word_to_id["<unk>"],
        word_to_id["<s>"],
        word_to_id["</s>"],
}

counts_flat = count_matrix.reshape(-1)
total_pairs = counts_flat.size

all_indices = jnp.arange(total_pairs)
w1_all = all_indices // V
w2_all = all_indices % V

# Keep only positive-count bigrams whose both tokens are not special
valid_mask = (counts_flat > 0)
for sid in special_ids:
    valid_mask = valid_mask & (w1_all != sid) & (w2_all != sid)

# Invalid entries get sentinel -1 so they sink after sorting (we sort by -counts)
valid_counts = jnp.where(valid_mask, counts_flat, -1)

# Top-N indices by raw count (excluding special-token bigrams)
top_indices = jnp.argsort(-valid_counts)[:N]

records = []
for idx in top_indices:
    idx_int = int(idx)
    if idx_int < 0:
        continue
    c = int(counts_flat[idx_int])
    if c < 0:
        continue
    w1_id = idx_int // V
    w2_id = idx_int % V
    # Safety skip (in case)
    if w1_id in special_ids or w2_id in special_ids:
        continue
    w1 = id_to_word[w1_id]
    w2 = id_to_word[w2_id]
    p_smooth = float(prob_smoothed[w1_id, w2_id])
    records.append((w1, w2, c, p_smooth))

# Ensure exact descending order by count
records.sort(key=lambda r: r[3], reverse=True)
records = records[:N]

print(f"\nTop {N} most frequent bigram pairs (excluding <unk>, <s>, </s>):")
print(f"{'w1':<15} | {'w2':<15} | {'count':<8} | {'P_smooth':<10}")
print("-" * 60)
for w1, w2, c, p in records:
    print(f"{w1:<15} | {w2:<15} | {c:<8d} | {p:<10.6f}")


key = jr.PRNGKey(0)
num_sentences = 5
max_len = 40
avoid_unk = True

start_id = word_to_id["<s>"]
end_id = word_to_id["</s>"]
unk_id = word_to_id["<unk>"]

for s in range(num_sentences):
    prev = start_id
    out_tokens = []
    for t in range(max_len):
        row = prob_smoothed[prev]  # jnp vector
        if avoid_unk:
            # Zero `<unk>` then renormalize (only if more than one non-zero prob)
            row = row.at[unk_id].set(0.0)
            denom = row.sum()
            row = jnp.where(denom > 0, row / denom, row)
        key, subkey = jr.split(key)
        next_id = int(jr.categorical(subkey, jnp.log(row)))
        if next_id == end_id:
            break
        if next_id != start_id and next_id != end_id:
            out_tokens.append(id_to_word[int(next_id)])
        prev = next_id
    print(f"GEN {s+1}: {' '.join(out_tokens)}")
