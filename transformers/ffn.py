import jax
from flax import nnx


class FeedForwardNetwork(nnx.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, rngs: nnx.Rngs | None = None):
        self.lin1 = nnx.Linear(d_model, d_ff, rngs=rngs)
        self.lin2 = nnx.Linear(d_ff, d_model, rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, deterministic: bool = False):
        x = self.lin1(x)
        x = nnx.relu(x)
        x = self.dropout1(x, deterministic=deterministic)
        x = self.lin2(x)
        x = self.dropout2(x, deterministic=deterministic)
        return x


if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 32
    d_ff = 64
    dropout_rate = 0.1

    rng = jax.random.PRNGKey(0)
    x = jax.random.uniform(rng, (batch_size, seq_len, d_model))

    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, dropout=dropout_rate, rngs=nnx.Rngs(0))
    out_train = ffn(x)  # dropout active
    print("Train shape:", out_train.shape)

    out_eval = ffn(x, deterministic=True)
    print("Eval shape:", out_eval.shape)
