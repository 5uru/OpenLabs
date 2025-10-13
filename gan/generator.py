from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


class Generator(nnx.Module):
    def __init__(self, latent_dim: int = 100, base_channels: int = 256, rngs=nnx.Rngs(0)):
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # Project z -> 7*7*C, then reshape to (N, 7, 7, C)
        self.project = nnx.Linear(latent_dim, 7 * 7 * base_channels, rngs=rngs)

        # 7x7xC -> 14x14x(C/2) -> 28x28x(C/4)
        self.deconv1 = nnx.ConvTranspose(
                in_features=base_channels, out_features=base_channels // 2,
                kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.deconv2 = nnx.ConvTranspose(
                in_features=base_channels // 2, out_features=base_channels // 4,
                kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs
        )

        # Final RGB conv: 28x28x(C/4) -> 28x28x3
        self.to_rgb = nnx.Conv(
                in_features=base_channels // 4, out_features=3,
                kernel_size=(3, 3), padding="SAME", rngs=rngs
        )

    def __call__(self, z):
        # z: (N, latent_dim)
        n = z.shape[0]
        x = self.project(z)
        x = nnx.leaky_relu(x, 0.2)

        # Reshape to NHWC for ConvTranspose
        x = jnp.reshape(x, (n, 7, 7, self.base_channels))

        x = self.deconv1(x)  # Fixed: was using convT1 but defined as deconv1
        x = nnx.leaky_relu(x, 0.2)

        x = self.deconv2(x)  # Fixed: was using convT2 but defined as deconv2
        x = nnx.leaky_relu(x, 0.2)

        x = self.to_rgb(x)
        x = nnx.tanh(x)  # outputs in [-1, 1]
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (10, 100))  # (batch, latent_dim)

    model = Generator(rngs=nnx.Rngs(0))
    y = model(z)  # Fixed: was using x but should be z

    print("Input shape:", z.shape)
    print("Output shape:", y.shape)  # expected: (10, 28, 28, 3)

    params = nnx.state(model, nnx.Param)
    total_params = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print("Total parameters:", int(total_params))
