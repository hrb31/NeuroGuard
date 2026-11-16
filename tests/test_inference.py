"""Synthetic application checks; no MRI data, training or held-out evaluation."""
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from src.dataset import file_sha256
from src.inference import POINTS, Segmenter, prepare_scan, read_scan, render_prediction
from src.model import UNet
from src.preprocessing import preprocess


@pytest.fixture
def scan(tmp_path):
    image = np.random.default_rng(12).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    path = tmp_path / 'generated.tif'
    tifffile.imwrite(path, image, photometric='rgb')
    return path, image


@pytest.fixture
def engine(tmp_path, monkeypatch):
    # Test-generated, untrained weights are never accepted by the production hash
    # registry. This temporary patch is confined to pytest's fixture lifetime.
    torch.set_num_threads(2)
    model = UNet()
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.zero_()  # sigmoid exactly .5 tests the >= boundary
    path = tmp_path / 'synthetic.pt'
    torch.save({'model': model.state_dict()}, path)
    monkeypatch.setattr('src.inference.CHECKPOINT_SHA256',
                        {'bce': file_sha256(path), 'tversky': file_sha256(path)})
    return Segmenter(path, path), model.eval(), path


def test_original_preprocessing_and_flair_parity(scan):
    path, image = scan
    np.testing.assert_array_equal(read_scan(path), image)
    expected, _ = preprocess(image, np.zeros((256, 256), dtype=np.uint8),
                             size=256, channels=(0, 1, 2), training=False)
    np.testing.assert_array_equal(prepare_scan(image), expected)
    mask = np.zeros((256, 256), dtype=bool)
    mask[2:8, 3:10] = True
    flair, overlay, binary = render_prediction(image, mask)
    np.testing.assert_array_equal(flair, image[..., 1])
    np.testing.assert_array_equal(overlay[~mask], np.repeat(flair[..., None], 3, axis=2)[~mask])
    np.testing.assert_array_equal(binary > 0, mask)
    assert set(np.unique(binary)) == {0, 255}


@pytest.mark.parametrize('shape,dtype', [((255, 256, 3), 'uint8'),
    ((256, 256), 'uint8'), ((256, 256, 4), 'uint8'), ((256, 256, 3), 'float32'),
    ((256, 256, 3), 'uint16'), ((2, 256, 256, 3), 'uint8')])
def test_reject_wrong_shape_dtype_or_pages(tmp_path, shape, dtype):
    path = tmp_path / 'bad.tif'
    tifffile.imwrite(path, np.zeros(shape, dtype=dtype), photometric='rgb' if shape[-1] in (3, 4) else 'minisblack')
    with pytest.raises(ValueError, match='256'):
        read_scan(path)


def test_reject_corrupt_extension_and_oversize(tmp_path):
    for name, contents in [('image.png', b'not TIFF'), ('bad.tif', b'not TIFF'), ('empty.tif', b'')]:
        path = tmp_path / name
        path.write_bytes(contents)
        with pytest.raises(ValueError):
            read_scan(path)
    path = tmp_path / 'huge.tif'
    with path.open('wb') as stream:
        stream.truncate(4 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='4 MiB'):
        read_scan(path)


@pytest.mark.parametrize('choice', list(POINTS))
def test_frozen_points_strict_reload_and_exact_predictions(engine, scan, choice):
    segmenter, original, checkpoint = engine
    path, image = scan
    before = checkpoint.read_bytes()
    flair, outputs = segmenter.predict(path, choice, True)
    with torch.inference_mode():
        probability = original(torch.from_numpy(prepare_scan(image))[None]).sigmoid()[0, 0].numpy()
    for output, (arm, threshold) in zip(outputs, POINTS[choice], strict=True):
        np.testing.assert_array_equal(output['mask'] > 0, probability >= threshold)
        assert not segmenter.models[arm].training
        assert all(not p.requires_grad for p in segmenter.models[arm].parameters())
        if arm == 'tversky':
            assert 'failed (fallback)' in output['label']
    np.testing.assert_array_equal(flair, image[..., 1])
    assert checkpoint.read_bytes() == before


def test_no_unapproved_input_or_threshold(engine, scan):
    segmenter, _, _ = engine
    with pytest.raises(ValueError, match='Confirm'):
        segmenter.predict(scan[0], next(iter(POINTS)))
    with pytest.raises(ValueError, match='frozen'):
        segmenter.predict(scan[0], 'threshold .9', True)
    with pytest.raises(ValueError, match='Upload'):
        segmenter.predict(None, next(iter(POINTS)), True)


def test_missing_corrupt_and_wrong_architecture(tmp_path, scan, monkeypatch):
    path = tmp_path / 'bad.pt'
    segmenter = Segmenter(path, path)
    with pytest.raises(ValueError, match='unavailable'):
        segmenter.predict(scan[0], next(iter(POINTS)), True)
    path.write_bytes(b'untrusted pickle must never be loaded')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        segmenter.predict(scan[0], next(iter(POINTS)), True)
    model = UNet(base_channels=8)
    torch.save({'model': model.state_dict()}, path)
    monkeypatch.setattr('src.inference.CHECKPOINT_SHA256', {'bce': file_sha256(path)})
    with pytest.raises(RuntimeError, match='size mismatch'):
        segmenter.predict(scan[0], next(iter(POINTS)), True)


def test_nonfinite_weights_rejected(engine, scan, monkeypatch):
    segmenter, model, path = engine
    with torch.no_grad():
        model.output.bias.fill_(float('nan'))
    torch.save({'model': model.state_dict()}, path)
    monkeypatch.setattr('src.inference.CHECKPOINT_SHA256', {'bce': file_sha256(path)})
    with pytest.raises(ValueError, match='nonfinite'):
        segmenter.predict(scan[0], next(iter(POINTS)), True)


def test_gradio_callback_clears_failure_and_renders_predictions(engine, scan):
    gr = pytest.importorskip('gradio')
    from app import build_app
    demo = build_app(engine[0])
    callback = next(fn.fn for fn in demo.fns.values() if fn.api_name == 'segment')
    result = callback(str(scan[0]), next(iter(POINTS)), True)
    assert result[0].shape == (256, 256)
    assert len(result[1]) == len(result[2]) == 1
    assert 'Complete' in result[3]
    failed = callback(str(scan[0]), next(iter(POINTS)), False)
    assert failed[:3] == (None, [], [])
    assert 'Confirm' in failed[3]
    assert demo.analytics_enabled is False
    assert all(component.format == 'png' for component in demo.blocks.values()
               if isinstance(component, (gr.Image, gr.Gallery)))
    assert all('share' not in component.buttons for component in demo.blocks.values()
               if isinstance(component, (gr.Image, gr.Gallery)))
