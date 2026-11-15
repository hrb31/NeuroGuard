import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
torch = pytest.importorskip('torch')

from src.data import audit_dataset, digest, freeze_split
from src.dataset import MRIDataset, load_manifest, make_loader, verify_dataset
from src.model import UNet
from src.smoke import tiny_overfit
from src.training import evaluate, fit, load_checkpoint


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def development(tmp_path):
    root = tmp_path/'raw'
    for i in range(20):
        patient = f'TCGA_CS_{i:04d}'
        folder = root/patient
        folder.mkdir(parents=True)
        for s in (1, 2):
            mask = np.zeros((32, 32), np.uint8)
            if s == 1:
                mask[8:24, 8:24] = 255
            rng = np.random.default_rng(i*10+s)
            image = rng.integers(1, 30, (32, 32, 3), dtype=np.uint8)
            image[mask > 0] += 150
            tifffile.imwrite(folder/f'{patient}_{s}.tif', image, photometric='rgb')
            tifffile.imwrite(folder/f'{patient}_{s}_mask.tif', mask)
    path = tmp_path/'manifest.json'
    manifest = freeze_split(root, audit_dataset(root), path, channel_review='Synthetic test only')
    return root, manifest, path


def test_manifest_verification_and_isolation(development):
    root, manifest, path = development
    before = path.read_bytes()
    loaded = load_manifest(path, manifest['manifest_sha256'])
    verify_dataset(root, loaded, expected_shape=(32, 32, 3))
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='identity'):
        load_manifest(path, 'wrong')
    with pytest.raises(ValueError, match='only'):
        MRIDataset(root, loaded, 'test')
    train, val = [MRIDataset(root, loaded, s, size=32) for s in ('train', 'validation')]
    assert not {r['patient'] for r in train.records} & {r['patient'] for r in val.records}
    a, b = train[0], train[0]
    torch.testing.assert_close(a['image'], b['image'], rtol=0, atol=0)
    assert a['mask'].shape == (1, 32, 32)
    assert set(a['mask'].unique().tolist()) <= {0, 1}
    original = val[0]['image'].clone()
    val.epoch = 99
    torch.testing.assert_close(val[0]['image'], original, rtol=0, atol=0)
    # Recompute self-hash but retain external identity: assignments still cannot change.
    changed = copy.deepcopy(loaded)
    changed['records'][0]['split'] = 'validation' if changed['records'][0]['split'] != 'validation' else 'train'
    changed['manifest_sha256'] = digest({k:v for k,v in changed.items() if k != 'manifest_sha256'})
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        load_manifest(path, manifest['manifest_sha256'])


def test_changed_bytes_and_inventory_rejected(development):
    root, manifest, _ = development
    row = manifest['records'][0]
    path = root/row['mask']
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='bytes'):
        verify_dataset(root, manifest, expected_shape=None)
    tifffile.imwrite(root/'extra.tif', np.zeros((32, 32), np.uint8))
    with pytest.raises(ValueError, match='inventory'):
        verify_dataset(root, manifest, expected_shape=None)


@pytest.mark.parametrize('batch,height,width', [(1, 256, 256), (2, 64, 64), (1, 65, 67)])
def test_unet_forward_backward(batch, height, width):
    model = UNet()
    x = torch.randn(batch, 3, height, width)
    logits = model(x)
    assert logits.shape == (batch, 1, height, width)
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(3, height, width))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No local CUDA GPU')
def test_cuda_forward_backward():
    model = UNet().cuda()
    output = model(torch.randn(2, 3, 64, 64, device='cuda'))
    output.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def small_config(experiment='bce'):
    config = json.loads(Path(f'configs/{experiment}.json').read_text())
    config.update(max_epochs=2, batch_size=14, num_workers=0)
    config['model']['base_channels'] = 2
    config['preprocessing']['size'] = 32
    return config


def test_loader_seeds_workers_and_shuffle(development):
    root, manifest, _ = development
    ds = MRIDataset(root, manifest, 'train', size=32)
    config = small_config()
    config['num_workers'] = 2
    a = next(iter(make_loader(ds, config, 3)))
    b = next(iter(make_loader(ds, config, 3)))
    torch.testing.assert_close(a['image'], b['image'], rtol=0, atol=0)
    torch.testing.assert_close(a['index'], b['index'], rtol=0, atol=0)


def test_baseline_and_validation_outputs(development):
    root, manifest, _ = development
    ds = MRIDataset(root, manifest, 'validation', size=32)
    loader = make_loader(ds, small_config())
    baseline = evaluate(None, loader, 'cpu', manifest['size_cutoffs'])['reports'][.5]
    assert baseline['positive_patient_macro']['dice'] == 0
    assert baseline['positive_patient_macro']['recall'] == 0
    assert baseline['empty']['false_positive_rate'] == 0
    model_report = evaluate(UNet(base_channels=2), loader, 'cpu', manifest['size_cutoffs'],
                            (.3, .5, .7), torch.nn.BCEWithLogitsLoss())
    assert np.isfinite(model_report['loss']) and len(model_report['reports']) == 3


@pytest.mark.parametrize('interruption', ['epoch_boundary', 'mid_epoch', 'before_first_checkpoint'])
@pytest.mark.parametrize('experiment', ['bce', 'tversky'])
def test_checkpoint_resume_matches_uninterrupted(development, tmp_path, monkeypatch, interruption, experiment):
    import src.training as training
    root, manifest, _ = development
    config = small_config(experiment)
    def datasets():
        return [MRIDataset(root, manifest, s, size=32) for s in ('train', 'validation')]
    identity = {'config': config, 'manifest_sha256': manifest['manifest_sha256'],
                'commit': 'synthetic', 'environment': 'CPU test', 'size_cutoffs': manifest['size_cutoffs']}
    full_model, full_history = fit(*datasets(), config, tmp_path/'full', identity)
    original = training.save_checkpoint
    original_train = training.train_epoch
    calls = 0
    def interrupted_train(*args):
        nonlocal calls
        calls += 1
        result = original_train(*args)
        if (interruption == 'mid_epoch' and calls == 2) or (interruption == 'before_first_checkpoint' and calls == 1):
            raise InterruptedError('Synthetic disconnect before second epoch checkpoint')
        return result
    def interrupted(directory, state):
        original(directory, state)
        if interruption == 'epoch_boundary' and state['epoch'] == 1:
            raise InterruptedError('Synthetic disconnect after completed epoch')
    monkeypatch.setattr(training, 'save_checkpoint', interrupted)
    monkeypatch.setattr(training, 'train_epoch', interrupted_train)
    with pytest.raises(InterruptedError):
        fit(*datasets(), config, tmp_path/'resumed', identity)
    monkeypatch.setattr(training, 'save_checkpoint', original)
    monkeypatch.setattr(training, 'train_epoch', original_train)
    resumed_model, resumed_history = fit(*datasets(), config, tmp_path/'resumed', identity, resume=True)
    for a, b in zip(full_history, resumed_history):
        assert a['train_loss'] == b['train_loss']
        assert a['validation'] == b['validation']
    for key, tensor in full_model.state_dict().items():
        torch.testing.assert_close(tensor, resumed_model.state_dict()[key], rtol=0, atol=0)
    assert (tmp_path/'resumed'/'best.pt').exists()
    state = load_checkpoint(tmp_path/'resumed', identity)
    assert state['epoch'] == 2
    other_config = small_config('tversky' if experiment == 'bce' else 'bce')
    with pytest.raises(ValueError, match='differs'):
        load_checkpoint(tmp_path/'resumed', dict(identity, config=other_config))
    with pytest.raises(ValueError, match='differs'):
        load_checkpoint(tmp_path/'resumed', dict(identity, commit='changed'))
    pointer = json.loads((tmp_path/'resumed'/'latest.json').read_text())
    file = tmp_path/'resumed'/pointer['file']
    file.write_bytes(file.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(tmp_path/'resumed', identity)


def test_best_checkpoint_and_early_stop_are_distinct(development, tmp_path, monkeypatch):
    import src.training as training
    root, manifest, _ = development
    config = small_config()
    config.update(max_epochs=10, patience=2)
    identity = {'config': config, 'manifest_sha256': manifest['manifest_sha256'],
                'size_cutoffs': manifest['size_cutoffs']}
    datasets = [MRIDataset(root, manifest, s, size=32) for s in ('train', 'validation')]
    scores = iter([.2, .20005, .20006])
    def fake_train(model, *args):
        with torch.no_grad():
            model.output.bias.fill_(next(scores))
        return 1.0
    def fake_validation(model, *args):
        score = float(model.output.bias.item())
        return {'loss': 1.0, 'reports': {.5: {'positive_patient_macro': {'dice': score}}}}
    monkeypatch.setattr(training, 'train_epoch', fake_train)
    monkeypatch.setattr(training, 'evaluate', fake_validation)
    model, history = fit(*datasets, config, tmp_path/'stopping', identity)
    assert len(history) == 3
    assert model.output.bias.item() == pytest.approx(.20006)
    state = load_checkpoint(tmp_path/'stopping', identity)
    assert state['bad_epochs'] == 2 and state['epoch'] == 3
    # Resuming early-stopped run must not grant another epoch.
    _, resumed = fit(*datasets, config, tmp_path/'stopping', identity, resume=True)
    assert len(resumed) == 3


def test_fit_rejects_wrong_roles_and_identity(development, tmp_path):
    root, manifest, _ = development
    train, val = [MRIDataset(root, manifest, s, size=32) for s in ('train', 'validation')]
    config = small_config()
    identity = {'config': config, 'manifest_sha256': manifest['manifest_sha256'],
                'size_cutoffs': manifest['size_cutoffs']}
    with pytest.raises(ValueError, match='roles'):
        fit(val, train, config, tmp_path/'bad-role', identity)
    with pytest.raises(ValueError, match='identity'):
        fit(train, val, config, tmp_path/'bad-hash', dict(identity, manifest_sha256='changed'))
    val.size = 64
    with pytest.raises(ValueError, match='preprocessing'):
        fit(train, val, config, tmp_path/'bad-preprocessing', identity)


def test_private_validation_figures(development, tmp_path):
    from PIL import Image
    from src.reporting import save_curves, save_predictions
    root, manifest, _ = development
    validation = MRIDataset(root, manifest, 'validation', size=32)
    save_curves([{'epoch': 1, 'train_loss': .5, 'validation_loss': .6,
                  'validation': {'positive_patient_macro': {'dice': .2}}}], tmp_path)
    save_predictions(UNet(base_channels=2), validation, 'cpu', .5, tmp_path/'figures')
    paths = list(tmp_path.rglob('*.png'))
    assert len(paths) >= 2
    for path in paths:
        with Image.open(path) as image:
            image.verify()


@pytest.mark.parametrize('experiment', ['bce', 'tversky'])
def test_tiny_training_overfit(development, tmp_path, experiment):
    root, manifest, _ = development
    result = tiny_overfit(MRIDataset(root, manifest, 'train', size=32),
                          small_config(experiment), tmp_path/'smoke', steps=40)
    assert result['loss_decreased'] and result['checkpoint_roundtrip']
    assert result['debug_only']


@pytest.mark.parametrize('name', ['02_train_bce', '03_train_tversky'])
def test_training_notebook_is_clean_compilable_and_protects_test(name):
    nb = json.loads(Path(f'notebooks/{name}.ipynb').read_text(encoding='utf-8'))
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'training notebook cell {i}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None
    source = '\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'freeze_split(' not in source and 'split_patients(' not in source
    assert 'load_manifest(' in source and 'SMOKE_REVIEWED' in source
