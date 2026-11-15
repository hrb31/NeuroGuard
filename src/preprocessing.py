"""Paired 2D preprocessing, identical for both future loss experiments."""
import numpy as np
from PIL import Image


def preprocess(image, mask, *, size=256, channels=None, training=False, rng=None):
    image = np.asarray(image)
    mask = np.asarray(mask)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError("Expected aligned HWC image and HW mask")
    if not np.isfinite(image).all() or not np.isfinite(mask).all():
        raise ValueError("Non-finite input")
    if not (set(np.unique(mask)) <= {0, 1} or set(np.unique(mask)) <= {0, 255}):
        raise ValueError("Mask must be binary")
    if size <= 0:
        raise ValueError("Size must be positive")
    if channels is not None:
        if not channels or any(c < 0 or c >= image.shape[-1] for c in channels):
            raise ValueError("Invalid channels")
        image = image[..., list(channels)]
    planes = [Image.fromarray(image[..., c].astype(np.float32)) for c in range(image.shape[-1])]
    mask_image = Image.fromarray((mask > 0).astype(np.uint8))
    # Preserve aspect ratio with centred zero padding before resizing.
    h, w = mask.shape
    side = max(h, w)
    offset = ((side-w)//2, (side-h)//2)
    padded = []
    for plane in planes:
        square = Image.new("F", (side, side), 0)
        square.paste(plane, offset)
        padded.append(square.resize((size, size), Image.Resampling.BILINEAR))
    square_mask = Image.new("L", (side, side), 0)
    square_mask.paste(mask_image, offset)
    mask_image = square_mask.resize((size, size), Image.Resampling.NEAREST)
    if training:
        if rng is None:
            raise ValueError("Training augmentation requires an explicitly seeded RNG")
        if rng.random() < .5:
            padded = [p.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for p in padded]
            mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if rng.random() < .5:
            angle = float(rng.uniform(-10, 10))
            padded = [p.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0) for p in padded]
            mask_image = mask_image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    x = np.stack([np.asarray(p, dtype=np.float32) for p in padded], axis=0)
    for plane in x:
        nonzero = plane != 0
        if nonzero.any():
            values = plane[nonzero]
            std = float(values.std())
            plane[nonzero] = (values - values.mean()) / std if std > 1e-8 else 0
    return x, (np.asarray(mask_image) > 0).astype(np.float32)[None]
