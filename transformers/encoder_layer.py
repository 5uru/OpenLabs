from jax import numpy as jnp
import jax
from flax import nnx
from multi_head_attention import MultiHeadAttention
from ffn import FeedForwardNetwork


class EncoderLayer(nnx.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rngs: nnx.Rngs | None = None):
        # Multi-head self-attention
        self.mha = MultiHeadAttention(d_model, num_heads, rngs=rngs)

        # layer normalization
        self.norm1 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout, rngs=rngs)

        # Dropout layers
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, mask=None, deterministic: bool = False):
        # Multi-head self-attention
        attn_output = self.mha(x, x, x, mask=mask)
        attn_output = self.dropout1(attn_output, deterministic=deterministic)
        out1 = self.norm1(x + attn_output)  # Residual connection + LayerNorm

        # Feed-forward network
        ffn_output = self.ffn(out1, deterministic=deterministic)
        ffn_output = self.dropout2(ffn_output, deterministic=deterministic)
        out2 = self.norm2(out1 + ffn_output)  # Residual connection + LayerNorm

        return out2


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4
    d_ff = 64
    dropout_rate = 0.1

    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, (batch_size, seq_len, d_model))

    encoder_layer = EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout_rate, rngs=nnx.Rngs(0))
    out_train = encoder_layer(x)  # dropout active
    print("Train shape:", out_train.shape)

    out_eval = encoder_layer(x, deterministic=True)
    print("Eval shape:", out_eval.shape)