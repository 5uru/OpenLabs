from jax import numpy as jnp
import jax
from flax import nnx
from multi_head_attention import MultiHeadAttention
from ffn import FeedForwardNetwork

class DecoderLayer(nnx.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, rngs: nnx.Rngs | None = None):
        # Masked Multi-head attention
        self.mha1 = MultiHeadAttention(d_model, num_heads, rngs=rngs)
        # Multi-head attention over encoder output
        self.mha2 = MultiHeadAttention(d_model, num_heads, rngs=rngs)
        # Layer normalization
        self.norm1 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.norm2 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        self.norm3 = nnx.LayerNorm(d_model, dtype=jnp.float32, rngs=rngs)
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout=dropout, rngs=rngs)
        # Dropout layers
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)
    def __call__(self, x, encoder_output, src_mask=None, tgt_mask=None, deterministic: bool = False):
        # Masked Multi-head attention
        attn1 = self.mha1(x, x, x, mask=tgt_mask)
        attn1 = self.dropout1(attn1, deterministic=deterministic)
        out1 = self.norm1(x + attn1)  # Residual connection + LayerNorm

        # Multi-head attention over encoder output
        attn2 = self.mha2(out1, encoder_output, encoder_output, mask=src_mask)
        attn2 = self.dropout2(attn2, deterministic=deterministic)
        out2 = self.norm2(out1 + attn2)  # Residual connection + LayerNorm

        # Feed-forward network
        ffn_output = self.ffn(out2, deterministic=deterministic)
        ffn_output = self.dropout3(ffn_output, deterministic=deterministic)
        out3 = self.norm3(out2 + ffn_output)  # Residual connection + LayerNorm

        return out3
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 32
    num_heads = 4
    d_ff = 64
    dropout_rate = 0.1

    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, (batch_size, seq_len, d_model))
    encoder_output = jax.random.uniform(rng, (batch_size, seq_len, d_model))

    decoder_layer = DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout_rate, rngs=nnx.Rngs(0))
    out_train = decoder_layer(x, encoder_output)  # dropout active
    print("Train shape:", out_train.shape)

    out_eval = decoder_layer(x, encoder_output, deterministic=True)
    print("Eval shape:", out_eval.shape)