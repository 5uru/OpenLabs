from jax import numpy as jnp
import jax
from flax import nnx

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (batch_size, num_heads, seq_len_q, d_k)
        key: (batch_size, num_heads, seq_len_k, d_k)
        value: (batch_size, num_heads, seq_len_v, d_v)
        mask: boolean mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
              True = keep, False = mask out
    """
    d_k = query.shape[-1]
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)

    if mask is not None:
        # Ensure mask is bool; set masked positions to large negative
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

    attn_weights = nnx.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, value)
    return output


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model, num_heads, rngs: nnx.Rngs):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Note: use integer division //

        # Create the learnable projection matrices
        self.W_q = nnx.Linear(d_model, d_model, rngs=rngs) #think why we are doing from d_model -> d_model
        self.W_k = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_v = nnx.Linear(d_model, d_model, rngs=rngs)
        self.W_o = nnx.Linear(d_model, d_model, rngs=rngs)

    def __call__(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # Linear projections
        query = self.W_q(query)  # (batch_size, seq_len, d_model)
        key = self.W_k(key)      # (batch_size, seq_len, d_model)
        value = self.W_v(value)  # (batch_size, seq_len, d_model)

        # Reshape and transpose for multi-head attention
        def reshape_for_heads(x):
            x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
            return jnp.transpose(x, (0, 2, 1, 3))  # (batch_size, num_heads, seq_len, d_k)

        query = reshape_for_heads(query)
        key = reshape_for_heads(key)
        value = reshape_for_heads(value)

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(query, key, value, mask)  # (batch_size, num_heads, seq_len, d_k)

        # Concatenate heads and put through the final linear layer
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, d_k)
        attn_output = attn_output.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)

        output = self.W_o(attn_output)  # (batch_size, seq_len, d_model)
        return output

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4

    rng = jax.random.PRNGKey(0)
    rng, q_key, k_key, v_key = jax.random.split(rng, 4)

    query = jax.random.uniform(q_key, (batch_size, seq_len, d_model))
    key = jax.random.uniform(k_key, (batch_size, seq_len, d_model))
    value = jax.random.uniform(v_key, (batch_size, seq_len, d_model))

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, rngs=nnx.Rngs(0))
    output = mha(query, key, value)
    print("Output shape:", output.shape)