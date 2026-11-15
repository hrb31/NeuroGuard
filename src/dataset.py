"""Read-only frozen-manifest validation and deterministic development loaders."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import digest, read_pair, summarize, validate_split
from .preprocessing import preprocess


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Manifest path escapes canonical data root')
    return path


def load_manifest(path, expected_hash):
    """Verify against the original Phase 2 printed identity, not just self-hash."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    content = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if not expected_hash or digest(content) != expected_hash or manifest['manifest_sha256'] != expected_hash:
        raise ValueError('Frozen manifest identity mismatch')
    original = [{k: v for k, v in r.items() if k != 'split'} for r in manifest['records']]
    if digest(original) != manifest['dataset_fingerprint']:
        raise ValueError('Dataset fingerprint mismatch')
    validate_split(original, manifest['records'])
    if manifest['seed'] != 42 or not manifest.get('channel_review', '').strip():
        raise ValueError('Expected reviewed seed42 Phase 2 manifest')
    cutoffs = manifest['size_cutoffs']
    if len(cutoffs) != 2 or not 0 <= cutoffs[0] <= cutoffs[1] <= 1:
        raise ValueError('Invalid frozen size cutoffs')
    for split in ('train', 'validation', 'test'):
        if summarize([r for r in manifest['records'] if r['split'] == split]) != manifest['summary'][split]:
            raise ValueError('Frozen split summary mismatch')
    return manifest


def verify_dataset(root, manifest, *, expected_shape=(256, 256, 3)):
    """Hash/pair/dimension integrity only; never expose test pixels to a model.

    Canonical root resolver checks the complete TIFF inventory independently.
    Decode development pairs; test files receive byte-integrity checking only.
    """
    expected_files = {r[k] for r in manifest['records'] for k in ('image', 'mask')}
    actual_files = {p.relative_to(root).as_posix() for p in Path(root).rglob('*')
                    if p.suffix.lower() in {'.tif', '.tiff'}}
    if expected_files != actual_files:
        raise ValueError('TIFF inventory differs from frozen manifest')
    for row in manifest['records']:
        for kind in ('image', 'mask'):
            if file_sha256(safe_path(root, row[kind])) != row[kind+'_sha256']:
                raise ValueError('Frozen image/mask bytes changed')
        if expected_shape is not None and tuple(row['shape']) != tuple(expected_shape):
            raise ValueError('Unexpected frozen image dimensions')
        if row['split'] != 'test':
            image, mask = read_pair(root, row)
            if list(image.shape) != row['shape'] or image.shape[:2] != mask.shape:
                raise ValueError('Frozen pair dimensions changed')
            preprocess(image, mask, size=32)  # finite/binary integrity, not a new audit/split
            if int(np.count_nonzero(mask)) != row['foreground']:
                raise ValueError('Frozen foreground count changed')


class MRIDataset(Dataset):
    def __init__(self, root, manifest, split, *, size=256, channels=(0, 1, 2), seed=42):
        if split not in ('train', 'validation'):
            raise ValueError('Phase 3 exposes training and validation only')
        self.root, self.split = Path(root), split
        self.manifest_sha256 = manifest['manifest_sha256']
        self.records = [r for r in manifest['records'] if r['split'] == split]
        if not self.records:
            raise ValueError('Empty development subset')
        self.size, self.channels, self.seed, self.epoch = size, channels, seed, 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image, mask = read_pair(self.root, row)
        # Per-sample/epoch RNG makes augmentation independent of worker scheduling.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        x, y = preprocess(image, mask, size=self.size, channels=self.channels,
                          training=self.split == 'train', rng=rng)
        return {'image': torch.from_numpy(x.copy()), 'mask': torch.from_numpy(y.copy()),
                'patient': row['patient'], 'area_fraction': row['area_fraction'], 'index': index}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, epoch=0):
    dataset.epoch = epoch
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    return DataLoader(dataset, batch_size=config['batch_size'],
                      shuffle=dataset.split == 'train', num_workers=config['num_workers'],
                      worker_init_fn=seed_worker, generator=generator, drop_last=False,
                      persistent_workers=False, pin_memory=torch.cuda.is_available())
