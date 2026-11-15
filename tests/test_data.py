import copy
import json
import shutil
import numpy as np
import pytest
import tifffile

from src.data import audit_dataset, freeze_split, split_patients, validate_split
from src.preprocessing import preprocess
from src.visualization import save_previews
from src.acquisition import resolve_dataset_root


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "raw"
    for i in range(20):
        patient = f"TCGA_CS_{i:04d}"
        folder = root / patient
        folder.mkdir(parents=True)
        for s in (1, 2):
            rng = np.random.default_rng(i*10+s)
            image = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
            mask = np.zeros((12, 16), dtype=np.uint8)
            if s == 1:
                mask[3:7, 4:8] = 255
            tifffile.imwrite(folder / f"{patient}_{s}.tif", image, photometric="rgb")
            tifffile.imwrite(folder / f"{patient}_{s}_mask.tif", mask)
    return root


def test_actual_inventory_and_disjoint_split(dataset):
    audit = audit_dataset(dataset)
    assert audit["errors"] == []
    assert audit["summary"]["patients"] == 20
    assert audit["summary"]["positive_slices"] == audit["summary"]["empty_slices"] == 20
    rows = audit["records"]
    assigned = split_patients(rows)
    assert assigned == split_patients(rows)
    assert assigned != split_patients(rows, seed=43)
    sets = [{r["patient"] for r in assigned if r["split"] == s}
            for s in ("train", "validation", "test")]
    assert list(map(len, sets)) == [14, 3, 3]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == {r["patient"] for r in rows}
    assert {(r["image"], r["mask"]) for r in rows} == {(r["image"], r["mask"]) for r in assigned}


@pytest.mark.parametrize("fault", ["leak", "missing", "repeat", "mask_swap", "invalid"])
def test_reject_invalid_splits(dataset, fault):
    rows = audit_dataset(dataset)["records"]
    assigned = copy.deepcopy(split_patients(rows))
    if fault == "leak":
        assigned[0]["split"] = "test" if assigned[0]["split"] != "test" else "train"
    elif fault == "missing":
        assigned.pop()
    elif fault == "repeat":
        assigned[1] = assigned[0].copy()
    elif fault == "mask_swap":
        assigned[0]["mask"], assigned[1]["mask"] = assigned[1]["mask"], assigned[0]["mask"]
    else:
        assigned[0]["split"] = "other"
    with pytest.raises(ValueError):
        validate_split(rows, assigned)


@pytest.mark.parametrize("fault", ["missing", "orphan", "corrupt", "shape", "labels"])
def test_audit_rejects_bad_files(dataset, fault):
    mask = sorted(dataset.rglob("*_mask.tif"))[0]
    if fault == "missing":
        mask.unlink()
    elif fault == "orphan":
        tifffile.imwrite(mask.with_name("orphan_mask.tif"), np.zeros((12, 16), np.uint8))
    elif fault == "corrupt":
        mask.write_bytes(b"not a tiff")
    elif fault == "shape":
        tifffile.imwrite(mask, np.zeros((3, 4), np.uint8))
    else:
        tifffile.imwrite(mask, np.full((12, 16), 7, np.uint8))
    assert audit_dataset(dataset)["errors"]


def test_duplicate_detection(dataset):
    images = sorted(p for p in dataset.rglob("*.tif") if "_mask" not in p.stem)
    images[2].write_bytes(images[0].read_bytes())
    audit = audit_dataset(dataset)
    assert audit["cross_patient_duplicates"]
    with pytest.raises(ValueError, match="duplicates"):
        freeze_split(dataset, audit, dataset.parent / "manifest.json", channel_review="reviewed")
    with pytest.raises(ValueError, match="Nonblank"):
        freeze_split(dataset, audit, dataset.parent / "manifest.json", channel_review="reviewed", duplicate_review="inspected")


def test_freeze_idempotence_and_modification_guard(dataset):
    audit = audit_dataset(dataset)
    output = dataset.parent / "manifest.json"
    manifest = freeze_split(dataset, audit, output, channel_review="synthetic three channels")
    assert manifest == freeze_split(dataset, audit, output, channel_review="synthetic three channels")
    assert json.loads(output.read_text()) == manifest
    expected = np.quantile([r["area_fraction"] for r in manifest["records"]
                            if r["split"] == "train" and r["foreground"]], [1/3, 2/3])
    np.testing.assert_allclose(manifest["size_cutoffs"], expected)
    with pytest.raises(ValueError, match="differs"):
        freeze_split(dataset, audit, output, channel_review="synthetic three channels", seed=43)
    # Directory traversal order is filesystem-dependent. Rewriting an already
    # empty mask with identical bytes is a no-op, not a dataset modification.
    empty_row = next(row for row in audit["records"] if row["foreground"] == 0)
    empty_path = dataset / empty_row["mask"]
    empty_mask = tifffile.imread(empty_path)
    assert not np.count_nonzero(empty_mask)
    tifffile.imwrite(empty_path, np.zeros_like(empty_mask))
    unchanged = audit_dataset(dataset)
    assert unchanged["dataset_fingerprint"] == audit["dataset_fingerprint"]
    assert manifest == freeze_split(dataset, audit, output, channel_review="synthetic three channels")

    # Choose a known positive record from the sorted audit, then truly change it.
    positive_row = next(row for row in audit["records"] if row["foreground"] > 0)
    mask_path = dataset / positive_row["mask"]
    original_mask = tifffile.imread(mask_path)
    assert np.count_nonzero(original_mask) > 0
    tifffile.imwrite(mask_path, np.zeros_like(original_mask))
    modified_mask = tifffile.imread(mask_path)
    assert not np.array_equal(original_mask, modified_mask)
    assert np.count_nonzero(modified_mask) == 0
    changed = audit_dataset(dataset)
    assert changed["errors"] == []
    changed_row = next(row for row in changed["records"] if row["mask"] == positive_row["mask"])
    assert changed_row["mask_sha256"] != positive_row["mask_sha256"]
    assert changed["dataset_fingerprint"] != audit["dataset_fingerprint"]
    with pytest.raises(ValueError, match="changed"):
        freeze_split(dataset, audit, output, channel_review="synthetic three channels")
    assert json.loads(output.read_text()) == manifest


def test_preprocessing_alignment_and_train_only_augmentation():
    mask = np.zeros((32, 32), np.uint8)
    mask[5:15, 4:12] = 1
    image = mask[..., None].astype(np.float32) * np.arange(1, 33)[None, :, None]
    x, y = preprocess(image, mask, size=32)
    assert x.shape == (1, 32, 32) and y.shape == (1, 32, 32)
    np.testing.assert_array_equal(y[0], mask)
    np.testing.assert_array_equal(x[0] != 0, mask > 0)
    a = preprocess(image, mask, size=32, training=True, rng=np.random.default_rng(2))
    b = preprocess(image, mask, size=32, training=True, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert set(np.unique(a[1])) <= {0, 1}
    assert np.isfinite(a[0]).all()
    # Evaluation does not draw random augmentation even when a generator is passed.
    np.testing.assert_array_equal(preprocess(image, mask, size=32, rng=np.random.default_rng(3))[0], x)
    with pytest.raises(ValueError):
        preprocess(image, mask, training=True)


def test_no_cohort_normalisation_and_constant_channels():
    image = np.zeros((10, 20, 3), np.uint16)
    image[2:8, 4:12, 0] = 10
    image[2:8, 4:12, 1] = np.arange(48).reshape(6, 8)
    x, y = preprocess(image, np.zeros((10, 20), np.uint8), size=32, channels=(0, 1))
    assert x.shape == (2, 32, 32) and not y.any() and np.isfinite(x).all()
    z, _ = preprocess(np.zeros_like(image), np.zeros((10, 20), np.uint8))
    assert not z.any()


def test_flip_is_paired():
    class FlipOnly:
        def __init__(self):
            self.draws = iter([.1, .9])

        def random(self):
            return next(self.draws)
    mask = np.zeros((16, 16), np.uint8)
    mask[3:10, 2:6] = 1
    image = mask[..., None] * np.arange(1, 17)[None, :, None]
    x, y = preprocess(image, mask, size=16, training=True, rng=FlipOnly())
    np.testing.assert_array_equal(y[0], mask[:, ::-1])
    np.testing.assert_array_equal(x[0] != 0, mask[:, ::-1] > 0)


def test_notebook_code_is_valid_python_and_has_no_saved_outputs():
    from pathlib import Path
    nb = json.loads(Path('notebooks/01_data_exploration.ipynb').read_text())
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'notebook cell {index}', 'exec')
            assert cell['execution_count'] is None
            assert cell['outputs'] == []


def test_training_preview_outputs(dataset, tmp_path):
    assigned = split_patients(audit_dataset(dataset)["records"])
    paths = save_previews(dataset, assigned, tmp_path / "figures")
    assert len(paths) == 6
    from PIL import Image
    for path in paths:
        with Image.open(path) as image:
            image.verify()


def duplicate_layout(dataset, destination):
    """Reproduce the two observed Kaggle hierarchies using synthetic data only."""
    first = destination / "kaggle_3m"
    second = destination / "lgg-mri-segmentation" / "kaggle_3m"
    shutil.copytree(dataset, first)
    shutil.copytree(dataset, second)
    return first, second


@pytest.mark.parametrize("layout", ["patient_root", "wrapper", "named_root"])
def test_resolve_single_dataset_root(dataset, tmp_path, layout):
    supplied = dataset
    expected = dataset
    if layout != "patient_root":
        expected = tmp_path / "download" / "kaggle_3m"
        shutil.copytree(dataset, expected)
        supplied = expected.parent if layout == "wrapper" else expected
    root, report = resolve_dataset_root(supplied)
    assert root == expected.resolve()
    assert report["equivalent_copies"] == 1
    assert report["candidates"][0]["tiff_files"] == 80
    assert audit_dataset(root)["dataset_fingerprint"] == audit_dataset(dataset)["dataset_fingerprint"]


def test_identical_directory_copies_use_one_root_end_to_end(dataset, tmp_path):
    wrapper = tmp_path / "download"
    first, second = duplicate_layout(dataset, wrapper)
    # Demonstrate the original failure without disabling its integrity checks.
    doubled = audit_dataset(wrapper)
    assert doubled["summary"]["valid_pairs"] == 80
    assert len(doubled["duplicate_images"]) == 40
    assert not doubled["cross_patient_duplicates"]
    assert "Repeated patient/slice identifiers" in doubled["errors"]

    root, report = resolve_dataset_root(wrapper)
    assert root == first.resolve()
    assert report["equivalent_copies"] == 2
    assert [c["relative_root"] for c in report["candidates"]] == [
        "kaggle_3m", "lgg-mri-segmentation/kaggle_3m"]
    a, b = report["candidates"]
    assert a["files_sha256"] == b["files_sha256"]
    assert len(a["files_sha256"]) == 80  # Both image and mask files, not images alone.
    baseline = audit_dataset(dataset)
    audit = audit_dataset(root)
    assert audit["errors"] == []
    assert audit["summary"]["valid_pairs"] == 40
    assert audit["summary"]["positive_slices"] == audit["summary"]["empty_slices"] == 20
    assert audit["dataset_fingerprint"] == baseline["dataset_fingerprint"]
    manifest = freeze_split(root, audit, tmp_path / "manifest.json", channel_review="synthetic")
    assert manifest["records"] == split_patients(baseline["records"])
    assert manifest == freeze_split(root, audit, tmp_path / "manifest.json", channel_review="synthetic")
    assert len(save_previews(root, manifest["records"], tmp_path / "figures")) == 6
    # Resolution/audit/preprocessing/splitting must leave both copies untouched.
    assert resolve_dataset_root(wrapper)[1] == report
    assert audit_dataset(second)["dataset_fingerprint"] == baseline["dataset_fingerprint"]


@pytest.mark.parametrize("conflict", ["image", "mask", "missing", "extra", "renamed", "empty"])
def test_conflicting_directory_copies_are_rejected(dataset, tmp_path, conflict):
    wrapper = tmp_path / "download"
    first, second = duplicate_layout(dataset, wrapper)
    image = sorted(p for p in second.rglob("*.tif") if not p.stem.endswith("_mask"))[0]
    mask = image.with_name(image.stem + "_mask.tif")
    if conflict == "image":
        pixels = tifffile.imread(image)
        pixels[0, 0, 0] ^= 1
        tifffile.imwrite(image, pixels, photometric="rgb")
    elif conflict == "mask":
        pixels = tifffile.imread(mask)
        pixels[0, 0] = 255
        tifffile.imwrite(mask, pixels)
    elif conflict == "missing":
        mask.unlink()
    elif conflict == "extra":
        shutil.copyfile(mask, mask.with_name("extra_mask.tif"))
    elif conflict == "renamed":
        mask.rename(mask.with_name("renamed_mask.tif"))
    else:
        (wrapper / "empty_copy" / "kaggle_3m").mkdir(parents=True)
    before = {p.relative_to(wrapper).as_posix(): p.read_bytes()
              for p in wrapper.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="conflicting copies"):
        resolve_dataset_root(wrapper)
    assert before == {p.relative_to(wrapper).as_posix(): p.read_bytes()
                      for p in wrapper.rglob("*") if p.is_file()}
    assert audit_dataset(first)["errors"] == []


def test_resolution_keeps_duplicate_checks_inside_canonical_root(dataset, tmp_path):
    images = sorted(p for p in dataset.rglob("*.tif") if not p.stem.endswith("_mask"))
    images[2].write_bytes(images[0].read_bytes())  # Nonblank cross-patient duplicate.
    wrapper = tmp_path / "download"
    duplicate_layout(dataset, wrapper)
    root, _ = resolve_dataset_root(wrapper)
    audit = audit_dataset(root)
    assert audit["duplicate_images"] and audit["cross_patient_duplicates"]
    with pytest.raises(ValueError, match="Nonblank"):
        freeze_split(root, audit, tmp_path / "manifest.json", channel_review="reviewed",
                     duplicate_review="must not bypass")


def test_resolution_rejects_unassigned_tiffs(dataset, tmp_path):
    wrapper = tmp_path / "download"
    duplicate_layout(dataset, wrapper)
    tifffile.imwrite(wrapper / "unexpected.tif", np.zeros((12, 16), np.uint8))
    with pytest.raises(ValueError, match="outside candidate roots"):
        resolve_dataset_root(wrapper)


def test_resolution_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError, match="no LGG TIFF hierarchy"):
        resolve_dataset_root(tmp_path)


def test_resolution_rejects_overlapping_roots(dataset, tmp_path):
    parent = tmp_path / "kaggle_3m"
    shutil.copytree(dataset, parent)
    shutil.copytree(dataset, parent / "nested" / "kaggle_3m")
    with pytest.raises(ValueError, match="overlapping candidate roots"):
        resolve_dataset_root(parent)
