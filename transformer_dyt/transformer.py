import jax.numpy as jnp
from flax import nnx
import jax
import numpy as np
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from pe import PositionalEmbedding



class TransformerModel(nnx.Module):
    def __init__(self, sequence_length: int, vocab_size: int, embed_dim: int, latent_dim: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.encoder = TransformerEncoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.positional_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim, rngs=rngs)
        self.decoder = TransformerDecoder(embed_dim, latent_dim, num_heads, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dense = nnx.Linear(embed_dim, vocab_size, rngs=rngs)

    def __call__(self, encoder_inputs: jnp.array, decoder_inputs: jnp.array, mask: jnp.array = None, deterministic: bool = False):
        x = self.positional_embedding(encoder_inputs)
        encoder_outputs = self.encoder(x, mask=mask)

        x = self.positional_embedding(decoder_inputs)
        decoder_outputs = self.decoder(x, encoder_outputs, mask=mask)
        # per nnx.Dropout - disable (deterministic=True) for eval, keep (False) for training
        decoder_outputs = self.dropout(decoder_outputs, deterministic=deterministic)

        logits = self.dense(decoder_outputs)
        return logits

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    encoder_input = jax.random.randint(key, (10, 16), 0, 100)  # (batch_size, seq_length)
    decoder_input = jax.random.randint(key, (10, 16), 0, 100)  # (batch_size, seq_length)

    model = TransformerModel(sequence_length=16, vocab_size=100, embed_dim=32, latent_dim=64, num_heads=4, dropout_rate=0.1, rngs=nnx.Rngs(0))
    logits = model(encoder_input, decoder_input, deterministic=True)

    print("Encoder Input shape:", encoder_input.shape)
    print("Decoder Input shape:", decoder_input.shape)
    print("Logits shape:", logits.shape)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)