import jax.numpy as jnp
from flax import nnx
import jax
import numpy as np
from dyt import DyT


class TransformerDecoder(nnx.Module):
    def __init__(self, embed_dim: int, latent_dim: int, num_heads: int, rngs: nnx.Rngs, **kwargs):
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = nnx.MultiHeadAttention(num_heads=num_heads,
                                                  in_features=embed_dim,
                                                  decode=False,
                                                  rngs=rngs)
        self.attention_2 = nnx.MultiHeadAttention(num_heads=num_heads,
                                                  in_features=embed_dim,
                                                  decode=False,
                                                  rngs=rngs)

        self.dense_proj = nnx.Sequential(
                nnx.Linear(embed_dim, latent_dim, rngs=rngs),
                nnx.relu,
                nnx.Linear(latent_dim, embed_dim, rngs=rngs),
        )
        self.dyt_1 = DyT(embed_dim)
        self.dyt_2 = DyT(embed_dim)
        self.dyt_3 = DyT(embed_dim)

    def __call__(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs.shape[1])
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
            padding_mask = jnp.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None
        attention_output_1 = self.attention_1(
                inputs_q=inputs, inputs_v=inputs, inputs_k=inputs,  mask=causal_mask
        )
        out_1 = self.dyt_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2( ## https://github.com/google/flax/blob/main/flax/nnx/nn/attention.py#L403-L405
                inputs_q=out_1,
                inputs_v=encoder_outputs,
                inputs_k=encoder_outputs,
                mask=padding_mask,
        )
        out_2 = self.dyt_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.dyt_3(out_2 + proj_output)

    def get_causal_attention_mask(self, sequence_length):
        i = jnp.arange(sequence_length)[:, None]
        j = jnp.arange(sequence_length)
        mask = (i >= j).astype(jnp.int32)
        mask = jnp.reshape(mask, (1, 1, sequence_length, sequence_length))
        return mask

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 16, 32))  # (batch_size, seq_length, embed_dim)
    enc_out = jax.random.normal(key, (10, 16, 32))  # (batch_size, seq_length, embed_dim)

    model = TransformerDecoder(embed_dim=32, latent_dim=64, num_heads=4, rngs=nnx.Rngs(0))
    y = model(x, enc_out)

    print("Input shape:", x.shape)
    print("Encoder output shape:", enc_out.shape)
    print("Output shape:", y.shape)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)