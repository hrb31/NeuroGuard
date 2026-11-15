"""Dataset integrity audit and reproducible patient splits; no learning code."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import tifffile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def read_pair(root, row):
    image = tifffile.imread(Path(root) / row["image"])
    mask = tifffile.imread(Path(root) / row["mask"])
    if image.ndim == 2:
        image = image[..., None]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return image, mask


def audit_dataset(root):
    """Inspect every TIFF; retain errors instead of silently skipping bad pairs."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff"})
    errors, rows = [], []
    used_masks = set()
    masks = {p for p in files if p.stem.endswith("_mask")}
    images = [p for p in files if p not in masks]
    for image_path in images:
        candidates = [p for p in masks if p.parent == image_path.parent
                      and p.stem == image_path.stem + "_mask"]
        if len(candidates) != 1:
            errors.append(f"Expected exactly one mask: {image_path.relative_to(root)}")
            continue
        mask_path = candidates[0]
        used_masks.add(mask_path)
        match = re.fullmatch(r"(TCGA_[A-Za-z0-9_-]+)_(\d+)", image_path.stem)
        if not match or match[1] != image_path.parent.name:
            errors.append(f"Unexpected patient/filename convention: {image_path.relative_to(root)}")
            continue
        row = {"patient": match[1], "slice": int(match[2]),
               "image": image_path.relative_to(root).as_posix(),
               "mask": mask_path.relative_to(root).as_posix()}
        try:
            image, mask = read_pair(root, row)
            if image.ndim != 3 or image.shape[-1] not in (1, 3):
                raise ValueError(f"Unexpected image shape {image.shape}")
            if mask.ndim != 2 or image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions differ: {image.shape}/{mask.shape}")
            if not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError("Non-finite pixels")
            values = np.unique(mask).tolist()
            if not (set(values) <= {0, 1} or set(values) <= {0, 255}):
                raise ValueError(f"Non-binary mask values: {values}")
            foreground = int(np.count_nonzero(mask))
            row.update(shape=list(image.shape), image_dtype=str(image.dtype),
                       mask_dtype=str(mask.dtype), mask_values=values,
                       channel_min=image.min(axis=(0, 1)).tolist(),
                       channel_max=image.max(axis=(0, 1)).tolist(),
                       identical_channels=bool(np.all(image == image[..., :1])),
                       foreground=foreground, area_fraction=foreground / mask.size,
                       image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                       mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                       pixel_sha256=hashlib.sha256(
                           str((image.shape, str(image.dtype))).encode() + image.tobytes()
                       ).hexdigest())
            rows.append(row)
        except Exception as exc:
            errors.append(f"{row['image']}: {type(exc).__name__}: {exc}")
    errors.extend(f"Orphan mask: {p.relative_to(root)}" for p in sorted(masks - used_masks))
    if not rows:
        errors.append("No valid image-mask pairs")
    keys = [(r["patient"], r["slice"]) for r in rows]
    if len(keys) != len(set(keys)):
        errors.append("Repeated patient/slice identifiers")
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["pixel_sha256"]].append(row)
    duplicates = [[r["image"] for r in group] for group in by_hash.values() if len(group) > 1]
    cross_patient = [group for group in by_hash.values()
                     if len({r["patient"] for r in group}) > 1]
    summary = summarize(rows)
    summary.update(image_files=len(images), mask_files=len(masks), valid_pairs=len(rows),
                   dimensions=dict(Counter(str(r["shape"]) for r in rows)),
                   image_dtypes=dict(Counter(r["image_dtype"] for r in rows)),
                   mask_dtypes=dict(Counter(r["mask_dtype"] for r in rows)),
                   identical_channel_slices=sum(r["identical_channels"] for r in rows),
                   other_files=[p.relative_to(root).as_posix() for p in sorted(root.rglob("*"))
                                if p.is_file() and p not in files])
    return {"summary": summary, "errors": errors, "records": rows,
            "duplicate_images": duplicates,
            "cross_patient_duplicates": [[r["image"] for r in g] for g in cross_patient],
            "dataset_fingerprint": digest(rows),
            "channel_semantics": "Unknown from pixel values; check release documentation and review channels."}


def summarize(rows):
    patients = {r["patient"] for r in rows}
    positive_patients = {r["patient"] for r in rows if r["foreground"] > 0}
    empty_patients = {r["patient"] for r in rows if r["foreground"] == 0}
    return {"patients": len(patients), "slices": len(rows),
            "positive_slices": sum(r["foreground"] > 0 for r in rows),
            "empty_slices": sum(r["foreground"] == 0 for r in rows),
            "patients_with_positive": len(positive_patients),
            "patients_with_empty": len(empty_patients),
            "empty_only_patients": len(patients - positive_patients)}


def split_patients(rows, seed=42):
    patients = sorted({r["patient"] for r in rows})
    if len(patients) < 7:
        raise ValueError("Too few patients for the proposed 70/15/15 split")
    random.Random(seed).shuffle(patients)
    n_train, n_val = int(.70 * len(patients)), int(.15 * len(patients))
    groups = {"train": patients[:n_train], "validation": patients[n_train:n_train+n_val],
              "test": patients[n_train+n_val:]}
    lookup = {p: name for name, ids in groups.items() for p in ids}
    result = [dict(row, split=lookup[row["patient"]]) for row in rows]
    validate_split(rows, result)
    return result


def validate_split(original, assigned):
    if len(original) != len(assigned):
        raise ValueError("Split lost or added records")
    if len({r["image"] for r in assigned}) != len(assigned):
        raise ValueError("Repeated image")
    if len({r["mask"] for r in assigned}) != len(assigned):
        raise ValueError("Repeated mask")
    stripped = [{k: v for k, v in r.items() if k != "split"} for r in assigned]
    if sorted(stripped, key=lambda r: r["image"]) != sorted(original, key=lambda r: r["image"]):
        raise ValueError("Pair identity or content changed")
    groups = defaultdict(set)
    for row in assigned:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        groups[row["patient"]].add(row["split"])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("Patient leakage")
    if {r["split"] for r in assigned} != {"train", "validation", "test"}:
        raise ValueError("Empty subset")


def freeze_split(root, audit, output, *, channel_review, duplicate_review="", seed=42):
    """Write once, rerun idempotently; requires explicit review of flagged duplicates."""
    if audit["errors"] or not channel_review.strip():
        raise ValueError("Resolve audit errors and record channel review before freezing")
    if audit["cross_patient_duplicates"] and not duplicate_review.strip():
        raise ValueError("Cross-patient duplicates need documented resolution/review")
    duplicate_paths = {p for group in audit["cross_patient_duplicates"] for p in group}
    if any(r["image"] in duplicate_paths and
           (any(v != 0 for v in r["channel_min"]) or any(v != 0 for v in r["channel_max"]) or r["foreground"])
           for r in audit["records"]):
        raise ValueError("Nonblank cross-patient duplicates require resolution and a fresh audit")
    # Re-audit now to detect deleted, modified or newly introduced files.
    fresh = audit_dataset(root)
    if fresh["errors"] or fresh["dataset_fingerprint"] != audit["dataset_fingerprint"]:
        raise ValueError("Dataset changed since audit")
    rows = split_patients(audit["records"], seed)
    summary = {s: summarize([r for r in rows if r["split"] == s])
               for s in ("train", "validation", "test")}
    if any(v["positive_slices"] == 0 or v["empty_slices"] == 0 for v in summary.values()):
        raise ValueError("A subset has no positive or empty slices; review suitability without model results")
    train_areas = [r["area_fraction"] for r in rows if r["split"] == "train" and r["foreground"] > 0]
    manifest = {"seed": seed, "proportions": [.70, .15, .15],
                "algorithm": "sorted IDs, Python random.Random shuffle, floor train/validation",
                "dataset_fingerprint": audit["dataset_fingerprint"], "records": rows,
                "summary": summary, "channel_review": channel_review,
                "duplicate_review": duplicate_review,
                "size_cutoffs": np.quantile(train_areas, [1/3, 2/3]).tolist()}
    manifest["manifest_sha256"] = digest(manifest)
    output = Path(output)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise ValueError("Existing frozen manifest differs; do not overwrite or silently re-split")
    write_json(output, manifest)
    return manifest
