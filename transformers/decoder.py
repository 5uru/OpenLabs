from jax import numpy as jnp
import jax
from flax import nnx
from decoder_layer import DecoderLayer
from positional_encoding import PositionalEncoding


class Decoder(nnx.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 num_layers=6,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000, rngs: nnx.Rngs | None = None):

        # Output embedding
        self.embedding = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.scale = jnp.sqrt(d_model)
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        # Decoder layers
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout=dropout, rngs=rngs) for _ in range(num_layers)]
        # Dropout layer
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, encoder_output, src_mask=None, tgt_mask=None, deterministic: bool = False):
        seq_len = x.shape[1]
        # Embedding and positional encoding
        x = self.embedding(x) * self.scale
        x = self.pos_encoding(x)
        x = self.dropout(x, deterministic=deterministic)

        # Pass through each decoder layer
        for dec_layer in self.dec_layers:
            x = dec_layer(x, encoder_output, src_mask, tgt_mask, deterministic=deterministic)

        return x  # (batch_size, seq_len, d_model)
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    vocab_size = 1000
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    dropout_rate = 0.1

    rng = jax.random.PRNGKey(0)
    x = jax.random.randint(rng, (batch_size, seq_len), 0, vocab_size)
    encoder_output = jax.random.uniform(rng, (batch_size, seq_len, d_model))

    decoder = Decoder(vocab_size=vocab_size, d_model=d_model, num_layers=num_layers,
                      num_heads=num_heads, d_ff=d_ff, dropout=dropout_rate,
                      max_seq_length=5000, rngs=nnx.Rngs(0))
    out_train = decoder(x, encoder_output)  # dropout active
    print("Train shape:", out_train.shape)

    out_eval = decoder(x, encoder_output, deterministic=True)
    print("Eval shape:", out_eval.shape)

