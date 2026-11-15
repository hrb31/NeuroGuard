"""Small metadata requests; full download is explicitly restricted to Colab."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import requests

DATASET = "mateuszbuda/lgg-mri-segmentation"
VERSION = 2
METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"


def resolve_dataset_root(download_root):
    """Compare every candidate TIFF tree before selecting one; never edit raw files.

    Candidates are kaggle_3m directories or parents of TCGA patient directories.
    Unrecognised TIFF locations and overlapping roots fail closed. Metadata files
    are retained but are not used to establish image/mask-copy equivalence.
    """
    download_root = Path(download_root).resolve()
    if not download_root.is_dir():
        raise ValueError(f"Dataset download directory does not exist: {download_root}")
    entries = sorted(download_root.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise ValueError("Dataset root integrity error: symbolic links require review")
    tiffs = [p for p in entries if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    candidates = {p for p in [download_root, *entries] if p.is_dir() and p.name == "kaggle_3m"}
    candidates.update(p.parent.parent for p in tiffs if p.parent.name.startswith("TCGA_"))
    if not candidates or not tiffs:
        raise ValueError("Dataset root integrity error: no LGG TIFF hierarchy found")
    candidates = sorted(candidates, key=lambda p: (len(p.relative_to(download_root).parts),
                                                  p.relative_to(download_root).as_posix()))
    for i, candidate in enumerate(candidates):
        if any(candidate in other.parents or other in candidate.parents for other in candidates[i+1:]):
            raise ValueError("Dataset root integrity error: overlapping candidate roots require review")
    inventories = {candidate: {} for candidate in candidates}
    for file in tiffs:
        owners = [candidate for candidate in candidates if candidate in file.parents]
        if len(owners) != 1:
            raise ValueError(f"Dataset root integrity error: TIFF outside candidate roots: {file.relative_to(download_root)}")
        candidate = owners[0]
        with file.open("rb") as stream:
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        inventories[candidate][file.relative_to(candidate).as_posix()] = sha256
    canonical = candidates[0]  # Shallowest, then lexical; independent of filesystem ordering.
    reference = inventories[canonical]
    for candidate in candidates[1:]:
        inventory = inventories[candidate]
        missing = sorted(reference.keys() - inventory.keys())
        extra = sorted(inventory.keys() - reference.keys())
        different = sorted(name for name in reference.keys() & inventory.keys()
                           if reference[name] != inventory[name])
        if missing or extra or different:
            raise ValueError(
                f"Dataset root integrity error: conflicting copies {canonical} and {candidate}; "
                f"missing={missing[:5]}, extra={extra[:5]}, hash_mismatches={different[:5]}. "
                "No copy selected; raw files unchanged."
            )
    return canonical, {
        "download_root": str(download_root), "canonical_root": str(canonical),
        "selection_rule": "shallowest candidate, then lexical; all TIFF inventories must match",
        "equivalent_copies": len(candidates),
        "candidates": [{"relative_root": p.relative_to(download_root).as_posix(),
                        "tiff_files": len(inventories[p]), "files_sha256": inventories[p]}
                       for p in candidates],
        "comparison_scope": "all TIFF relative filenames and SHA256 bytes; other metadata not compared",
    }


def get_metadata():
    response = requests.get(METADATA_URL, timeout=30)
    response.raise_for_status()
    info = response.json()
    if not info.get("currentVersionNumber") or not info.get("licenseName"):
        raise ValueError("Metadata lacks version/licence; inspect Kaggle in the browser")
    return {"dataset": DATASET, "pinned_version": VERSION,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "metadata_url": METADATA_URL, "metadata": info}


def download_dataset(*, terms_accepted=False):
    import importlib.util
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    if not in_colab or not Path("/content").is_dir():
        raise RuntimeError("Dataset downloads are permitted in Google Colab only")
    if not terms_accepted:
        raise ValueError("Read and accept dataset/source usage terms before downloading")
    import kagglehub
    # KaggleHub uses Colab KAGGLE_API_TOKEN secret automatically if granted access.
    return Path(kagglehub.dataset_download(f"{DATASET}/versions/{VERSION}"))
