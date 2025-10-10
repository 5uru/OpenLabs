from flax import nnx
import jax
import numpy as np

class Generator(nnx.Module):
    def __init__(self, rngs=nnx.Rngs(0)):

        self.linear1 = nnx.Linear(100, 256, rngs=rngs)

        self.linear2 = nnx.Linear(256, 512, rngs=rngs)

        self.linear3 = nnx.Linear(512, 1024, rngs=rngs)

        self.linear4 = nnx.Linear(1024, 2352, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)

        x = self.linear2(x)
        x = nnx.relu(x)

        x = self.linear3(x)
        x = nnx.relu(x)

        x = self.linear4(x)

        x = nnx.tanh(x)

        # Reshape the output from (batch_size, 784) to a (batch_size, 28, 28) matrix
        x = x = x.reshape((x.shape[0],28, 28, 3))

        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,100))  # (batch_size, seq_length, embed_dim)

    model = Generator(rngs=nnx.Rngs(0))
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)