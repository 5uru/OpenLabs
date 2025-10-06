import numpy as np
from jax import numpy as jnp
from flax import nnx
from torchvision.datasets import CIFAR10
import optax
from cnn import Net

# Hyperparameters
LEARNING_RATE = 5e-3
BATCH_SIZE = 256
EVAL_EVERY = 1
NUM_EPOCHS = 20
RNG_SEED = 0

def load_cifar10(root: str = "data"):
    """Load CIFAR-10 into contiguous jnp arrays (NHWC, float32 in [0,1])."""
    train_ds = CIFAR10(root, download=True, train=True)
    test_ds = CIFAR10(root, download=True, train=False)

    def to_array(ds):
        imgs = np.stack([np.asarray(img, dtype=np.float32) for img, _ in ds], axis=0) / 255.0
        labels = np.array([label for _, label in ds], dtype=np.int32)
        return jnp.asarray(imgs), jnp.asarray(labels)

    return (*to_array(train_ds), *to_array(test_ds))

def batch_iterator(images, labels, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
    """Yield mini-batches as dicts with keys 'image' and 'label'."""
    n = images.shape[0]
    idxs = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        sl = idxs[start:start + batch_size]
        yield {'image': images[sl], 'label': labels[sl]}

def loss_fn(model: Net, batch):
    logits, _ = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

@nnx.jit
def train_step(model: Net, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(model, grads)

@nnx.jit
def eval_step(model: Net, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])

def main():
    rng = np.random.default_rng(RNG_SEED)

    train_images, train_labels, test_images, test_labels = load_cifar10()

    print(f"Train: {train_images.shape} {train_labels.shape}")
    print(f"Test:  {test_images.shape} {test_labels.shape}")

    model = Net(rngs=nnx.Rngs(RNG_SEED))
    optimizer = nnx.Optimizer(
            model,
            optax.adamw(learning_rate=LEARNING_RATE),  # can swap to optax.sgd(LEARNING_RATE, momentum=0.9)
            wrt=nnx.Param
    )
    metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss'),
    )

    metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
    }

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        metrics.reset()
        for batch in batch_iterator(train_images, train_labels, BATCH_SIZE, rng, shuffle=True):
            train_step(model, optimizer, metrics, batch)
        train_vals = metrics.compute()

        # Evaluation
        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            metrics.reset()
            for batch in batch_iterator(test_images, test_labels, BATCH_SIZE, rng, shuffle=False):
                eval_step(model, metrics, batch)
            test_vals = metrics.compute()

            metrics_history['train_loss'].append(train_vals['loss'])
            metrics_history['train_accuracy'].append(train_vals['accuracy'])
            metrics_history['test_loss'].append(test_vals['loss'])
            metrics_history['test_accuracy'].append(test_vals['accuracy'])

            print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} "
                    f"train_loss={train_vals['loss']:.4f} train_acc={train_vals['accuracy']:.4f} "
                    f"test_loss={test_vals['loss']:.4f} test_acc={test_vals['accuracy']:.4f}"
            )

    return metrics_history, model

if __name__ == "__main__":
    main()