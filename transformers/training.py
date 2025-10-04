import jax
from jax import numpy as jnp
from flax import nnx
import optax
from typing import Tuple
from main import Transformer

# Hyperparameters
SRC_VOCAB = 100
TGT_VOCAB = 100
D_MODEL = 32
LR = 1e-3
LABEL_SMOOTH = 0.1
GRAD_CLIP_NORM = 1.0
NUM_STEPS = 10
BATCH = 2
SRC_LEN = 5
TGT_LEN = 6  # includes BOS for input and shifted target

# Model
model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        dropout=0.1,
        rngs=nnx.Rngs(0),
)

# Optimizer (with optional gradient clipping)
optimizer = nnx.Optimizer(
        model,
        optax.chain(
                optax.clip_by_global_norm(GRAD_CLIP_NORM),
                optax.adam(LR),
        ),
        wrt=nnx.Param,
)

def label_smoothing_loss(
        logits: jnp.ndarray,
        targets: jnp.ndarray,
        epsilon: float = LABEL_SMOOTH,
) -> jnp.ndarray:
    """Efficient label smoothing cross entropy."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)                      # [B, T, V]
    # Gather log p(y_true)
    true_logp = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)  # [B, T]
    nll = -true_logp
    # Cross entropy with uniform distribution
    uniform_ce = -jnp.mean(log_probs, axis=-1)  # [B, T]
    loss = (1.0 - epsilon) * nll + epsilon * uniform_ce
    return jnp.mean(loss)

@nnx.jit
def train_step(
        model: Transformer,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
        src: jnp.ndarray,
        tgt_inp: jnp.ndarray,
        tgt_out: jnp.ndarray,
) -> Tuple[jnp.ndarray, jax.Array]:
    """One training step."""
    # Advance rngs for dropout
    model.rngs = nnx.Rngs(rng)

    def loss_fn(m: Transformer):
        logits = m(src, tgt_inp, deterministic=False)  # [B, T, V]
        return label_smoothing_loss(logits, tgt_out, LABEL_SMOOTH)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss, jax.random.split(rng, 2)[0]

# Synthetic data
key = jax.random.PRNGKey(42)
key_src, key_tgt, key_loop = jax.random.split(key, 3)
X_src = jax.random.randint(key_src, (BATCH, SRC_LEN), 0, SRC_VOCAB)
X_tgt_full = jax.random.randint(key_tgt, (BATCH, TGT_LEN), 0, TGT_VOCAB)

# Training loop
for step in range(1, NUM_STEPS + 1):
    tgt_input = X_tgt_full[:, :-1]   # teacher forcing input
    tgt_target = X_tgt_full[:, 1:]   # shifted targets
    loss, key_loop = train_step(model, optimizer, key_loop, X_src, tgt_input, tgt_target)
    if step % 1 == 0:
        print(f"step={step} loss={float(loss):.4f}")
