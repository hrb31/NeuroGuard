"""Save deterministic training-only previews; never silently inspect test slices."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .data import read_pair
from .preprocessing import preprocess


def save_previews(root, assigned, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = sorted([r for r in assigned if r["split"] == "train"], key=lambda r: r["image"])
    positive = sorted([r for r in train if r["foreground"] > 0], key=lambda r: r["area_fraction"])
    empty = [r for r in train if r["foreground"] == 0]
    selected = ([positive[0], positive[-1]] if positive else []) + empty[:1]
    paths = []
    for i, row in enumerate(selected):
        image, mask = read_pair(root, row)
        channels = image.shape[-1]
        fig, axes = plt.subplots(1, channels + 2, figsize=(4*(channels+2), 4))
        for c in range(channels):
            axes[c].imshow(image[..., c], cmap="gray")
            axes[c].set_title(f"Stored channel {c} (semantics unconfirmed)")
        axes[-2].imshow(mask > 0, cmap="gray", vmin=0, vmax=1)
        axes[-2].set_title("Ground truth")
        axes[-1].imshow(image[..., -1], cmap="gray")
        axes[-1].imshow(np.ma.masked_where(mask == 0, mask > 0), cmap="autumn", alpha=.5, vmin=0, vmax=1)
        axes[-1].set_title("Last stored channel + annotation")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Training example {i}: {'positive' if row['foreground'] else 'empty'} mask")
        fig.tight_layout()
        path = output / f"raw_example_{i}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, training in zip(axes, [False, True]):
            x, y = preprocess(image, mask, training=training, rng=np.random.default_rng(42))
            ax.imshow(x[-1], cmap="gray")
            ax.imshow(np.ma.masked_where(y[0] == 0, y[0]), cmap="autumn", alpha=.5, vmin=0, vmax=1)
            ax.set_title("Training augmentation" if training else "Deterministic preprocessing")
            ax.axis("off")
        path = output / f"processed_example_{i}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))
    return paths
