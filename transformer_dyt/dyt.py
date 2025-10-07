import jax
import jax.numpy as jnp
from flax import nnx
import optax
import jax.random as random
import numpy as np


class DyT(nnx.Module):
    def __init__(self, num_features: int, alpha_init_value: float = 0.5):
        self.alpha = nnx.Param(jnp.array(alpha_init_value, dtype=jnp.float32))
        self.weight = nnx.Param(jnp.ones((num_features,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_features,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(self.alpha * x)
        return x * self.weight + self.bias


if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (10, 5))

    model = DyT(num_features=5)
    y = model(x)

    print("Input:", x)
    print("Output:", y)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit  # automatic state management for JAX transforms
    def train_step(model, optimizer, x, y):
        def loss_fn(model):
            y_pred = model(x)  # call methods directly
            return ((y_pred - y) ** 2).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)  # in-place updates

        return loss
    y_true = random.normal(key, (10, 5))
    for step in range(100):
        loss = train_step(model, optimizer, x, y_true)
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss}")