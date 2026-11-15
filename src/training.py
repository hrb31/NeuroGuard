"""BCE experiment loop, reusable epoch evaluation and verified epoch-boundary resume."""
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .data import write_json
from .dataset import file_sha256, make_loader
from .metrics import aggregate, metric_row
from .model import UNet


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer, criterion, device):
    if loader.dataset.split != 'train':
        raise ValueError('Training requires training records only')
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch['image'].to(device), batch['mask'].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError('Logit/mask shapes differ')
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
            raise FloatingPointError('Missing/nonfinite gradients')
        optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, cutoffs, thresholds=(.5,), criterion=None):
    if loader.dataset.split != 'validation':
        raise ValueError('Phase 3 evaluation is validation only')
    if model is not None:
        model.eval()
    rows = {float(t): [] for t in thresholds}
    total, count = 0.0, 0
    for batch in loader:
        x, y = batch['image'].to(device), batch['mask'].to(device)
        logits = model(x) if model is not None else None
        if logits is not None:
            if logits.shape != y.shape or not torch.isfinite(logits).all():
                raise ValueError('Invalid validation logits')
            if criterion is not None:
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite validation loss')
                total += loss.item()*len(x)
        probabilities = torch.sigmoid(logits).cpu().numpy() if logits is not None else None
        target = y.cpu().numpy()
        for threshold in rows:
            predictions = probabilities >= threshold if probabilities is not None else np.zeros_like(target)
            for i in range(len(x)):
                rows[threshold].append(metric_row(predictions[i], target[i], batch['patient'][i],
                    float(batch['area_fraction'][i]), cutoffs))
        count += len(x)
    return {'loss': total/count if model is not None and criterion is not None else None,
            'reports': {t: aggregate(records) for t, records in rows.items()}}


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    write_json(temporary, content)
    temporary.replace(path)


def save_checkpoint(directory, state):
    """Two rotating slots; pointer changes only after a complete hashed save."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pointer = directory / 'latest.json'
    previous = json.loads(pointer.read_text()) if pointer.exists() else None
    slot = 'state-b.pt' if previous and previous['file'] == 'state-a.pt' else 'state-a.pt'
    destination = directory / slot
    temporary = directory / (slot+'.tmp')
    torch.save(state, temporary)
    checksum = file_sha256(temporary)
    temporary.replace(destination)
    if previous:
        atomic_json(directory / 'previous.json', previous)
    atomic_json(pointer, {'file': slot, 'sha256': checksum, 'epoch': state['epoch']})


def load_checkpoint(directory, identity):
    directory = Path(directory)
    pointer = json.loads((directory/'latest.json').read_text())
    if pointer['file'] not in ('state-a.pt', 'state-b.pt'):
        raise ValueError('Invalid checkpoint slot')
    path = directory/pointer['file']
    if file_sha256(path) != pointer['sha256']:
        raise ValueError('Checkpoint checksum mismatch; preserve files and investigate')
    # Only the user's own trusted Drive run, never a downloaded/untrusted pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != identity:
        raise ValueError('Resume config/manifest/commit/environment differs')
    return state


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def fit(train_dataset, validation_dataset, config, directory, identity, *, resume=False):
    """Construct a fresh seeded model; resume replays only an interrupted epoch.

    Identity includes complete config, manifest hash, source commit and environment.
    Criterion is passed into the epoch helpers; Phase 4 can reuse them unchanged.
    """
    if identity['config'] != config or config['experiment'] != 'bce' or config['loss'] != 'BCEWithLogitsLoss(mean)':
        raise ValueError('Phase 3 supports the frozen BCE configuration only')
    if train_dataset.split != 'train' or validation_dataset.split != 'validation':
        raise ValueError('Incorrect development dataset roles')
    if {r['patient'] for r in train_dataset.records} & {r['patient'] for r in validation_dataset.records}:
        raise ValueError('Patient leakage between training and validation')
    for dataset in (train_dataset, validation_dataset):
        if dataset.manifest_sha256 != identity['manifest_sha256']:
            raise ValueError('Dataset does not match frozen experiment identity')
        if (dataset.size != config['preprocessing']['size'] or
            list(dataset.channels) != config['preprocessing']['channels'] or dataset.seed != config['seed']):
            raise ValueError('Dataset preprocessing differs from frozen config')
    seed_everything(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'],
                                 weight_decay=config['optimizer']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history, best, patience_reference, bad_epochs, start, best_state = [], -1.0, -1.0, 0, 0, None
    if resume and (directory/'latest.json').exists():
        state = load_checkpoint(directory, identity)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best, bad_epochs = state['history'], state['best'], state['bad_epochs']
        patience_reference, best_state = state['patience_reference'], state['best_model']
        start = state['epoch']
        restore_rng(state['rng'])
    elif resume:
        # A disconnect before the first pointer commit has no completed epoch.
        identity_path = directory/'identity.json'
        if not identity_path.exists() or json.loads(identity_path.read_text()) != identity:
            raise ValueError('No completed checkpoint and no matching run identity')
        print('No completed checkpoint: replaying epoch 1 with the original seed and budget.', flush=True)
    elif any(directory.iterdir()):
        raise ValueError('Run directory is nonempty; choose resume or a NEW run ID')
    atomic_json(directory/'identity.json', identity)
    for epoch in range(start, config['max_epochs']):
        if bad_epochs >= config['patience']:
            break
        began = time.monotonic()
        train_loss = train_epoch(model, make_loader(train_dataset, config, epoch), optimizer, criterion, device)
        validation = evaluate(model, make_loader(validation_dataset, config), device,
                              identity['size_cutoffs'], (config['checkpoint_threshold'],), criterion)
        score = validation['reports'][.5]['positive_patient_macro']['dice']
        if score is None:
            raise ValueError('No eligible validation positive patients')
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if score > patience_reference + config['min_delta']:
            patience_reference, bad_epochs = score, 0
        else:
            bad_epochs += 1
        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'validation_loss': validation['loss'],
                        'validation': validation['reports'][.5], 'seconds': time.monotonic()-began})
        save_checkpoint(directory, {'epoch': epoch+1, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'best_model': best_state, 'best': best,
            'bad_epochs': bad_epochs, 'patience_reference': patience_reference,
            'history': history, 'rng': rng_state(), 'identity': identity})
        atomic_json(directory/'history.json', history)
        print(f"Epoch {epoch+1}/{config['max_epochs']}: train={train_loss:.5f}, val={validation['loss']:.5f}, Dice={score:.5f}", flush=True)
    if best_state is None:
        raise ValueError('No completed training epoch')
    model.load_state_dict(best_state)
    temporary = directory/'best.pt.tmp'
    torch.save({'model': best_state, 'identity': identity, 'validation_dice': best}, temporary)
    temporary.replace(directory/'best.pt')
    atomic_json(directory/'best_checksum.json', {'sha256': file_sha256(directory/'best.pt')})
    atomic_json(directory/'history.json', history)
    return model, history
