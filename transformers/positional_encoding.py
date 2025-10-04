from jax import numpy as jnp
import jax
from flax import nnx

class PositionalEncoding(nnx.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        position = jnp.arange(max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
        pe = jnp.zeros((max_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = jnp.expand_dims(pe, axis=0)                         # (1, max_len, d_model)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len]

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 32
    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, (batch_size, seq_len, d_model))
    pos_enc = PositionalEncoding(d_model=d_model, max_len=5000)
    out = pos_enc(x)
    print("Output shape:", out.shape)
