from flax import nnx
import jax
import numpy as np

class Discriminator(nnx.Module):
    def __init__(self, rngs=nnx.Rngs(0)):

        self.conv1 = nnx.Conv(in_features=3, out_features=64, kernel_size=(3,3),
                                padding='SAME', rngs=rngs)

        self.conv2 = nnx.Conv(in_features=64, out_features=128, kernel_size=(3,3), strides=(2,2),
                                padding='SAME', rngs=rngs)


        self.conv3 = nnx.Conv(in_features=128, out_features=128, kernel_size=(3,3), strides=(2,2),
                                padding='SAME', rngs=rngs)
        self.conv4 = nnx.Conv(in_features=128, out_features=256, kernel_size=(3,3), strides=(2,2),
                              padding='SAME', rngs=rngs)

        self.dropout = nnx.Dropout(0.3, rngs=rngs)
        self.linear = nnx.Linear(4096, 1, rngs=rngs)


    def __call__(self, x, deterministic=False):

        x = self.conv1(x)
        x = nnx.leaky_relu(x, 0.2)

        x = self.conv2(x)
        x = nnx.leaky_relu(x, 0.2)

        x = self.conv3(x)
        x = nnx.leaky_relu(x, 0.2)

        x = self.conv4(x)
        x = nnx.leaky_relu(x, 0.2)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = self.dropout(x, deterministic=deterministic)
        x = self.linear(x)

        return nnx.sigmoid(x)

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
