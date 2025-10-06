import jax
import jax.numpy as jnp
from jax import grad
import jax.image as jimg
from flax import nnx
import numpy as np
import matplotlib.pyplot as plt
from cnn import Net
from training import load_cifar10, main


def _normalize_cam(cam: jnp.ndarray) -> jnp.ndarray:
    cam = jnp.maximum(cam, 0.0)
    denom = jnp.max(cam)
    cam = jnp.where(denom > 0, cam / (denom + 1e-8), cam)
    return cam


def grad_cam(model, x, target_class: int | None = None, training: bool = False):
    """
    Supports x shape (H,W,C) or (B,H,W,C).
    Returns (cams, pred_classes):
      cams: (B,H,W) jnp.ndarray
      pred_classes: (B,) numpy array
    """
    single = False
    if x.ndim == 3:
        x = x[None, ...]
        single = True

    # Forward to obtain logits + saved conv activations
    logits, conv_acts = model(x, training=training)  # conv_acts: (B,Hc,Wc,Cc)
    pred_classes = jnp.argmax(logits, axis=-1)
    if target_class is None:
        target_class = pred_classes  # shape (B,)
    else:
        target_class = jnp.full(pred_classes.shape, target_class, dtype=jnp.int32)

    def per_example(cam_feats, cls_idx):
        # cam_feats: (Hc,Wc,Cc)
        def get_target_logit(conv4_act):
            pooled = model.pool(conv4_act[None, ...])         # (1,Hp,Wp,Cc)
            flat = pooled.reshape(1, -1)
            local_logits = model.dense_layers(flat, training=training)
            return local_logits[0, cls_idx]

        g = grad(get_target_logit)(cam_feats)
        weights = jnp.mean(g, axis=(0, 1))                   # (Cc,)
        cam_map = jnp.tensordot(cam_feats, weights, axes=([-1], [0]))  # (Hc,Wc)
        return _normalize_cam(cam_map)

    cams = jax.vmap(per_example)(conv_acts, target_class)

    if single:
        return np.array(cams[0]), int(pred_classes[0])
    return np.array(cams), np.array(pred_classes)


def _resize_cam_to_image(cam: np.ndarray, img_hw: tuple[int, int]) -> np.ndarray:
    if cam.shape != img_hw:
        cam_j = jnp.array(cam)[None, ..., None]
        resized = jimg.resize(cam_j, (1, img_hw[0], img_hw[1], 1), method="bilinear")[0, ..., 0]
        return np.array(resized)
    return cam


def _first_indices_per_class(labels: np.ndarray, num_classes: int) -> np.ndarray:
    # Returns the first occurrence index for each class 0..num_classes-1
    idxs = [int(np.where(labels == c)[0][0]) for c in range(num_classes)]
    return np.array(idxs, dtype=np.int32)


def _plot_gradcams(model, images, labels, out_file: str, title_prefix: str):
    rows, cols = 2, images.shape[0]
    plt.figure(figsize=(cols * 1.9, rows * 2.0))
    for i in range(cols):
        img = images[i]
        true_label = int(labels[i])
        cam, pred = grad_cam(model, img, training=False)
        cam = _resize_cam_to_image(cam, img.shape[:2])

        # Original
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.array(img))
        plt.title(f"T:{true_label}")
        plt.axis("off")

        # Overlay
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(np.array(img), alpha=0.65)
        plt.imshow(cam, cmap="jet", alpha=0.35)
        plt.title(f"P:{pred}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    print(f"Saved to {out_file}")
    plt.close()


def main_visualization():
    rngs = nnx.Rngs(0)
    model = Net(rngs=rngs)
    train_imgs, train_labels, test_imgs, test_labels = load_cifar10()

    # Select one sample per class
    test_labels_np = np.array(test_labels)
    sel_idx = _first_indices_per_class(test_labels_np, num_classes=10)
    sample_images = test_imgs[sel_idx]
    sample_labels = test_labels[sel_idx]

    # Before training
    _plot_gradcams(model, sample_images, sample_labels,
                   out_file="true_and_gradcam.png",
                   title_prefix="Before")

    # Train (assumes training.main returns (state, trained_model))
    _, trained_model = main()

    # After training
    _plot_gradcams(trained_model, sample_images, sample_labels,
                   out_file="true_and_gradcam_after_training.png",
                   title_prefix="After")


if __name__ == "__main__":
    main_visualization()
