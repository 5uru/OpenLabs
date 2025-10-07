import requests
import zipfile
import tempfile
import pathlib
import random
import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx
import tiktoken
import grain.python as grain
import tqdm

from utils import format_dataset, CustomPreprocessing
from training import train_step, eval_step
from transformer import TransformerModel


def run():
    url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        zip_file_path = temp_path / "spa-eng.zip"

        response = requests.get(url)
        zip_file_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        text_file = temp_path / "spa-eng" / "spa.txt"
        with open(text_file) as f:
            lines = f.read().split("\n")[:-1]

    text_pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        spa = "[start] " + spa + " [end]"
        text_pairs.append((eng, spa))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]

    print(f"{len(text_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

    tokenizer = tiktoken.get_encoding("cl100k_base")

    vocab_size = tokenizer.n_vocab
    sequence_length = 20

    train_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in train_pairs]
    val_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in val_pairs]
    test_data = [format_dataset(eng, spa, tokenizer, sequence_length) for eng, spa in test_pairs]

    batch_size = 64

    train_sampler = grain.IndexSampler(
            len(train_data),
            shuffle=True,
            seed=12,
            shard_options=grain.NoSharding(),
            num_epochs=1,
    )

    val_sampler = grain.IndexSampler(
            len(val_data),
            shuffle=False,
            seed=12,
            shard_options=grain.NoSharding(),
            num_epochs=1,
    )

    train_loader = grain.DataLoader(
            data_source=train_data,
            sampler=train_sampler,
            worker_count=4,
            worker_buffer_size=2,
            operations=[
                    CustomPreprocessing(),
                    grain.Batch(batch_size=batch_size, drop_remainder=True),
            ],
    )

    val_loader = grain.DataLoader(
            data_source=val_data,
            sampler=val_sampler,
            worker_count=4,
            worker_buffer_size=2,
            operations=[
                    CustomPreprocessing(),
                    grain.Batch(batch_size=batch_size),
            ],
    )

    eval_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            accuracy=nnx.metrics.Accuracy(),
    )

    train_metrics_history = {"train_loss": []}
    eval_metrics_history = {"test_loss": [], "test_accuracy": []}

    rng = nnx.Rngs(0)
    embed_dim = 256
    latent_dim = 2048
    num_heads = 8
    dropout_rate = 0.5
    learning_rate = 1.5e-3
    num_epochs = 10

    bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"
    train_total_steps = len(train_data) // batch_size

    model = TransformerModel(sequence_length, vocab_size, embed_dim, latent_dim, num_heads, dropout_rate, rngs=rng)
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(learning_rate))

    def train_one_epoch(epoch):
        model.train()
        with tqdm.tqdm(
                desc=f"[train] epoch: {epoch}/{num_epochs}, ",
                total=train_total_steps,
                bar_format=bar_format,
                leave=True,
        ) as pbar:
            for batch in train_loader:
                loss = train_step(model, optimizer, batch)
                train_metrics_history["train_loss"].append(loss.item())
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

    def evaluate_model(epoch):
        model.eval()
        eval_metrics.reset()
        for val_batch in val_loader:
            eval_step(model, val_batch, eval_metrics)
        for metric, value in eval_metrics.compute().items():
            eval_metrics_history[f"test_{metric}"].append(value)
        print(f"[test] epoch: {epoch + 1}/{num_epochs}")
        print(f"- total loss: {eval_metrics_history['test_loss'][-1]:0.4f}")
        print(f"- Accuracy: {eval_metrics_history['test_accuracy'][-1]:0.4f}")

    for epoch in range(num_epochs):
        train_one_epoch(epoch)
        evaluate_model(epoch)


if __name__ == "__main__":
    # Optional explicit start method (macOS / Python 3.13)
    # import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    run()