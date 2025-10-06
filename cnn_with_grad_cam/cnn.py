import jax
import jax.numpy as jnp
from flax import nnx

class Net(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 128, kernel_size=5, padding=2, rngs=rngs)
        self.conv2 = nnx.Conv(128, 128, kernel_size=5, padding=2, rngs=rngs)
        self.conv3 = nnx.Conv(128, 256, kernel_size=3, padding=1, rngs=rngs)
        self.conv4 = nnx.Conv(256, 256, kernel_size=3, padding=1, rngs=rngs)

        self.bn_conv1 = nnx.BatchNorm(128, rngs=rngs)
        self.bn_conv2 = nnx.BatchNorm(128, rngs=rngs)
        self.bn_conv3 = nnx.BatchNorm(256, rngs=rngs)
        self.bn_conv4 = nnx.BatchNorm(256, rngs=rngs)
        # Removed axis_name (was causing unbound axis error)
        self.bn_dense1 = nnx.BatchNorm(1024, rngs=rngs)
        self.bn_dense2 = nnx.BatchNorm(512, rngs=rngs)

        self.dropout_conv = nnx.Dropout(rate=0.25, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

        self.fc1 = nnx.Linear(256 * 8 * 8, 1024, rngs=rngs)
        self.fc2 = nnx.Linear(1024, 512, rngs=rngs)
        self.fc3 = nnx.Linear(512, 10, rngs=rngs)

        self.pool = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    def conv_layers(self, x, training: bool = True):
        out = self.conv1(x)
        out = self.bn_conv1(out, use_running_average=not training)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out = self.bn_conv2(out, use_running_average=not training)
        out = jax.nn.relu(out)

        out = self.pool(out)
        out = self.dropout_conv(out, deterministic=not training)

        out = self.conv3(out)
        out = self.bn_conv3(out, use_running_average=not training)
        out = jax.nn.relu(out)

        out = self.conv4(out)
        out = self.bn_conv4(out, use_running_average=not training)
        out = jax.nn.relu(out)
        conv4_activations = out  # Save for Grad-CAM

        out = self.pool(out)
        out = self.dropout_conv(out, deterministic=not training)
        return out, conv4_activations

    def dense_layers(self, x, training: bool = True):
        out = self.fc1(x)
        out = self.bn_dense1(out, use_running_average=not training)
        out = jax.nn.relu(out)
        out = self.dropout(out, deterministic=not training)

        out = self.fc2(out)
        out = self.bn_dense2(out, use_running_average=not training)
        out = jax.nn.relu(out)
        out = self.dropout(out, deterministic=not training)

        out = self.fc3(out)
        return out

    def __call__(self, x, training: bool = True):
        out, conv4_activations = self.conv_layers(x, training=training)
        out = out.reshape(out.shape[0], -1)  # flatten
        out = self.dense_layers(out, training=training)
        return out, conv4_activations

if __name__ == "__main__":
    # Initialize model
    rngs = nnx.Rngs(0)
    model = Net(rngs=rngs)

    # Dummy input
    x = jnp.ones((4, 32, 32, 3))  # NHWC format (Flax uses channels-last)

    # Forward pass (training mode)
    logits, conv4_activations = model(x, training=True)

    # Forward pass (eval mode)
    logits_eval, _ = model(x, training=False)
    print("Logits shape (training):", logits.shape)
    print("Logits shape (eval):", logits_eval.shape)
    print("Conv4 activations shape:", conv4_activations.shape)