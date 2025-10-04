from jax import numpy as jnp

def create_padding_mask(seq):
    seq = jnp.array(seq)
    # True where padding (token id 0)
    mask = (seq == 0)
    return mask[:, jnp.newaxis, jnp.newaxis, :]  # (batch, 1, 1, seq_len)

def create_future_mask(size):
    # True where positions should be masked (future tokens)
    return jnp.triu(jnp.ones((1, 1, size, size), dtype=bool), k=1)

def create_masks(src, tgt):
    src = jnp.array(src)
    tgt = jnp.array(tgt)

    src_padding_mask = create_padding_mask(src)                 # (batch,1,1,src_len)
    tgt_padding_mask = create_padding_mask(tgt)                 # (batch,1,1,tgt_len)

    tgt_len = tgt.shape[1]
    future_mask = create_future_mask(tgt_len)                   # (1,1,tgt_len,tgt_len)

    # Broadcast padding mask over query length
    tgt_padding_broadcast = jnp.broadcast_to(
            tgt_padding_mask, (tgt.shape[0], 1, tgt_len, tgt_len)
    )                                                           # (batch,1,tgt_len,tgt_len)

    # Combined mask: True means "mask out"
    tgt_mask = jnp.logical_or(future_mask, tgt_padding_broadcast)

    return src_padding_mask, tgt_mask

if __name__ == "__main__":
    src = jnp.array([[7, 6, 0, 0, 0],
                     [1, 2, 3, 0, 0],
                     [4, 5, 6, 7, 8]])

    tgt = jnp.array([[1, 2, 3, 0],
                     [4, 5, 0, 0],
                     [6, 7, 8, 9]])

    src_mask, tgt_mask = create_masks(src, tgt)
    print("Source mask shape:", src_mask.shape)
    print("Target mask shape:", tgt_mask.shape)
    print("Source mask:\n", src_mask.astype(int))
    print("Target mask (1 means masked):\n", tgt_mask.astype(int))
