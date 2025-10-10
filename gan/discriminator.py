from flax import nnx
import jax
import numpy as np

class Discriminator(nnx.Module):
    def __init__(self, rngs=nnx.Rngs(0)):

        self.linear1 = nnx.Linear(2352, 1024, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.3, rngs=rngs)

        self.linear2 = nnx.Linear(1024, 512, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.3, rngs=rngs)

        self.linear3 = nnx.Linear(512, 256, rngs=rngs)
        self.dropout3 = nnx.Dropout(0.3, rngs=rngs)

        self.linear4 = nnx.Linear(256, 1, rngs=rngs)


    def __call__(self, x, deterministic=False):
        # We transform the input of (batch_size, 28, 2, 3) to (batch_size, 2352)
        x = x.reshape(x.shape[0], 2352)

        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout1(x, deterministic=deterministic)

        x = self.linear2(x)
        x = nnx.relu(x)
        x = self.dropout2(x, deterministic=deterministic)

        x = self.linear3(x)
        x = nnx.relu(x)
        x = self.dropout3(x, deterministic=deterministic)

        x = self.linear4(x)

        x = nnx.sigmoid(x)

        return x

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10,28, 28, 3))

    model = Discriminator(rngs=nnx.Rngs(0))
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = nnx.state(model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print("Total parameters:", total_params)
