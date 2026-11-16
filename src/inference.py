"""Single-image application inference; no dataset evaluation or metrics."""
from pathlib import Path
from threading import Lock

import numpy as np
import tifffile
import torch

from .dataset import file_sha256
from .model import UNet
from .preprocessing import preprocess


CHECKPOINT_SHA256 = {
    'bce': 'ba0ce014980ecac8e8670357f2f2eff2484c719e1c2a7080ec5367dc85701eda',
    'tversky': '1828ab28fe7077923447d6fe3f31a297173b14b93db6323ad68f9349bfe95708',
}
POINTS = {
    'BCE · 0.5 (common)': [('bce', .5)],
    'BCE · 0.8 (validation-selected)': [('bce', .8)],
    'Tversky · 0.5 (FP constraint failed; fallback)': [('tversky', .5)],
    'Compare · common thresholds': [('bce', .5), ('tversky', .5)],
    'Compare · validation-selected / fallback': [('bce', .8), ('tversky', .5)],
}
MAX_FILE_BYTES = 4 * 1024 * 1024


def read_scan(path):
    """Bound TIFF decoding before allocating pixels; never convert RGB uploads."""
    path = Path(path)
    if path.suffix.lower() not in {'.tif', '.tiff'}:
        raise ValueError('Upload the original LGG three-channel .tif or .tiff image, not a mask or screenshot.')
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError('The TIFF must be nonempty and no larger than 4 MiB.')
    try:
        with tifffile.TiffFile(path) as source:
            if (len(source.pages) != 1 or source.pages[0].shape != (256, 256, 3)
                    or source.pages[0].dtype != np.dtype('uint8')):
                raise ValueError('Expected one 256 × 256 × 3 uint8 TIFF in the original stored channel order.')
            image = source.pages[0].asarray()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Cannot decode this TIFF. Use an unmodified, compatible LGG image.') from exc
    return image


def prepare_scan(image):
    # A dummy empty mask satisfies the existing paired preprocessing API.
    # No ground truth is used and no accuracy metrics are computed.
    return preprocess(image, np.zeros((256, 256), dtype=np.uint8),
                      size=256, channels=(0, 1, 2), training=False)[0]


def render_prediction(image, mask):
    """Display raw stored FLAIR channel 1; colouring never affects prediction."""
    flair = image[..., 1].copy()
    overlay = np.repeat(flair[..., None], 3, axis=2)
    overlay[mask] = np.rint(.55 * overlay[mask] + .45 * np.array([40, 210, 155])).astype(np.uint8)
    return flair, overlay, mask.astype(np.uint8) * 255


class Segmenter:
    """Lazily cache checksum-verified original weights; CPU by default."""
    def __init__(self, bce_checkpoint, tversky_checkpoint, device='cpu'):
        if device not in {'cpu', 'cuda'}:
            raise ValueError('Device must be cpu or cuda.')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Start with --device cpu.')
        self.device = torch.device(device)
        self.paths = {'bce': Path(bce_checkpoint), 'tversky': Path(tversky_checkpoint)}
        self.models = {}
        self.lock = Lock()

    def _load(self, arm):
        if arm in self.models:
            return self.models[arm]
        path = self.paths[arm]
        if not path.is_file():
            raise ValueError(f'{arm.upper()} checkpoint unavailable. Configure its original best.pt path; see docs/demo.md.')
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError(f'{arm.upper()} checkpoint is unexpectedly large. Use the original best.pt.')
        if file_sha256(path) != CHECKPOINT_SHA256[arm]:
            raise ValueError(f'{arm.upper()} checkpoint SHA256 mismatch. Preserve the file and check the original Phase 5 identity.')
        state = torch.load(path, map_location='cpu', weights_only=True)
        model = UNet(in_channels=3, base_channels=16)
        model.load_state_dict(state['model'], strict=True)
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError(f'{arm.upper()} checkpoint contains nonfinite weights.')
        model.to(self.device).eval().requires_grad_(False)
        self.models[arm] = model
        return model

    def predict(self, path, choice, confirmed=False):
        if not confirmed:
            raise ValueError('Confirm this is a compatible, public, de-identified LGG research image first.')
        if choice not in POINTS:
            raise ValueError('Choose one of the documented frozen operating points.')
        if not path:
            raise ValueError('Upload a compatible TIFF image first.')
        image = read_scan(path)
        x = torch.from_numpy(prepare_scan(image).copy())[None].to(self.device)
        outputs = []
        with self.lock, torch.inference_mode():
            for arm, threshold in POINTS[choice]:
                probability = self._load(arm)(x).sigmoid()[0, 0].cpu().numpy()
                if probability.shape != (256, 256) or not np.isfinite(probability).all():
                    raise ValueError('The model produced an invalid output. Do not interpret this result.')
                mask = probability >= threshold  # frozen evaluation convention
                flair, overlay, binary = render_prediction(image, mask)
                label = f'{arm.upper()} · threshold {threshold:g}'
                if arm == 'tversky':
                    label += ' · FP constraint failed (fallback)'
                outputs.append({'label': label, 'overlay': overlay, 'mask': binary})
        return flair, outputs
