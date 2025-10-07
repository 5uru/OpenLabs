import jax.numpy as jnp
from flax import nnx
import jax
import numpy as np
from dyt import DyT

class TransformerEncoder(nnx.Module):
    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = nnx.MultiHeadAttention(num_heads=num_heads,
                                                in_features=embed_dim,
                                                decode=False,
                                                rngs=rngs)
        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, dense_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(dense_dim, embed_dim, rngs=rngs),
        )

        self.dyt_1 = DyT(embed_dim)
        self.dyt_2 = DyT(embed_dim)

    def __call__(self, inputs, mask=None):
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
        else:
            padding_mask = None

        attention_output = self.attention(
                inputs_q = inputs, inputs_k = inputs, inputs_v = inputs, mask=padding_mask, decode = False
        )
        proj_input = self.dyt_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.dyt_2(proj_input + proj_output)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 16, 32))  # (batch_size, seq_length, embed_dim)

    model = TransformerEncoder(embed_dim=32, dense_dim=64, num_heads=4, rngs=nnx.Rngs(0))
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)