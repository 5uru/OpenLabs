import jax
from flax import nnx
from encoder import Encoder
from decoder import Decoder
from utils import create_masks


class Transformer(nnx.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_layers=6,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000, rngs: nnx.Rngs | None = None):

        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length, rngs=rngs)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length, rngs=rngs)
        # in_features should be d_model
        self.final_layer = nnx.Linear(d_model, tgt_vocab_size, rngs=rngs)

    def __call__(self, src, tgt, deterministic: bool = False):
        src_mask, tgt_mask = create_masks(src, tgt)

        enc_output = self.encoder(src, mask=src_mask, deterministic=deterministic)
        dec_output = self.decoder(tgt, enc_output, src_mask=src_mask, tgt_mask=tgt_mask, deterministic=deterministic)

        final_output = self.final_layer(dec_output)
        return final_output


if __name__ == "__main__":
    batch_size = 2
    src_seq_len = 5
    tgt_seq_len = 6
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 64
    dropout_rate = 0.1

    rng = jax.random.PRNGKey(0)
    src = jax.random.randint(rng, (batch_size, src_seq_len), 0, src_vocab_size)
    tgt = jax.random.randint(rng, (batch_size, tgt_seq_len), 0, tgt_vocab_size)

    transformer = Transformer(src_vocab_size=src_vocab_size,
                              tgt_vocab_size=tgt_vocab_size,
                              d_model=d_model,
                              num_layers=num_layers,
                              num_heads=num_heads,
                              d_ff=d_ff,
                              dropout=dropout_rate,
                              rngs=nnx.Rngs(0))

    out_train = transformer(src, tgt)
    print("Train shape:", out_train.shape)

    out_eval = transformer(src, tgt, deterministic=True)
    print("Eval shape:", out_eval.shape)
