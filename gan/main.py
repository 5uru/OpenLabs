# python
from generator import Generator
from discriminator import Discriminator
from dataset_generator import main as generate_dataset
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt

# Initialize models with different RNG streams
generator = Generator(rngs=nnx.Rngs(0))
discriminator = Discriminator(rngs=nnx.Rngs(1))

# Optimizers with improved learning rates
lr_schedule = optax.exponential_decay(
        init_value=0.0001,  # Lower initial learning rate
        transition_steps=500,
        decay_rate=0.95,
        end_value=0.00001
)

optimizer_D = nnx.Optimizer(
        discriminator,
        optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=lr_schedule, b1=0.5)
        ),
        wrt=nnx.Param,
)

optimizer_G = nnx.Optimizer(
        generator,
        optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adamw(learning_rate=lr_schedule, b1=0.5)
        ),
        wrt=nnx.Param,
)

# Normalize data to [-1, 1] range
data = (jnp.array(generate_dataset(n=70000), dtype=jnp.float32) * 2.0) - 1.0

def generate_samples(generator, n=16):
    noise = jax.random.normal(jax.random.PRNGKey(42), (n, 100))
    samples = generator(noise)
    # Rescale from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    return samples

def discriminator_loss(model, imgs, labels):
    logits = model(imgs)
    # Binary cross entropy loss
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    # Minimal L2 regularization
    l2_penalty = 0.00001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    return loss + l2_penalty

def generator_loss(generator_model, noise):
    fake_imgs = generator_model(noise)
    logits_fake = discriminator(fake_imgs, deterministic=True)
    # Generator tries to make discriminator predict real (1)
    valid_labels = jnp.ones_like(logits_fake)
    return optax.sigmoid_binary_cross_entropy(logits_fake, valid_labels).mean()

grad_fn_d = nnx.value_and_grad(discriminator_loss)
grad_fn_g = nnx.value_and_grad(generator_loss)

# Training loop with improved stability
batch_size = 64  # Larger batch size for stability
total_samples = data.shape[0]
n_epochs = 1000  # Train longer

for epoch in range(n_epochs):
    # Shuffle data at the beginning of each epoch
    perm = jax.random.permutation(jax.random.PRNGKey(epoch), total_samples)
    shuffled_data = data[perm]

    # Track losses for reporting
    d_losses, g_losses = [], []

    for i in range(0, total_samples, batch_size):
        # Ensure we don't go out of bounds
        if i + batch_size > total_samples:
            batch = shuffled_data[i:total_samples]
            current_batch_size = total_samples - i
        else:
            batch = shuffled_data[i:i+batch_size]
            current_batch_size = batch_size

        real_imgs = batch

        # Train discriminator
        key = jax.random.PRNGKey(epoch * 10000 + i)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        # Generate fake images
        noise = jax.random.normal(subkey1, (current_batch_size, 100))
        fake_imgs = generator(noise)

        # Add small noise to labels for label smoothing
        real_labels = jnp.ones((current_batch_size, 1)) * 0.9  # Label smoothing
        fake_labels = jnp.zeros((current_batch_size, 1)) + 0.1  # Label smoothing

        # Train discriminator on real images
        loss_d_real, grads_d_real = grad_fn_d(discriminator, real_imgs, real_labels)

        # Train discriminator on fake images
        loss_d_fake, grads_d_fake = grad_fn_d(discriminator, fake_imgs, fake_labels)

        # Average gradients and update discriminator
        loss_d = (loss_d_real + loss_d_fake) / 2
        grads_d = jax.tree_util.tree_map(lambda g1, g2: (g1 + g2) / 2, grads_d_real, grads_d_fake)
        optimizer_D.update(discriminator, grads_d)

        # Train generator (once per iteration is typically enough)
        noise = jax.random.normal(subkey2, (current_batch_size, 100))
        loss_g, grads_g = grad_fn_g(generator, noise)
        optimizer_G.update(generator, grads_g)

        d_losses.append(loss_d)
        g_losses.append(loss_g)

    # Print average loss per epoch
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        avg_d_loss = sum(d_losses) / len(d_losses) if d_losses else float('nan')
        avg_g_loss = sum(g_losses) / len(g_losses) if g_losses else float('nan')
        print(f"Epoch {epoch}, Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}")

        # Generate and save sample images
        samples = generate_samples(generator)
        plt.figure(figsize=(8, 8))
        for i in range(min(16, samples.shape[0])):
            plt.subplot(4, 4, i+1)
            plt.imshow(samples[i])
            plt.axis('off')
        plt.savefig(f"samples_epoch_{epoch}.png")
        plt.close()

# Save final model samples
samples = generate_samples(generator, n=25)
plt.figure(figsize=(10, 10))
for i in range(min(25, samples.shape[0])):
    plt.subplot(5, 5, i+1)
    plt.imshow(samples[i])
    plt.axis('off')
plt.savefig("final_samples.png")
plt.close()



