"""Small metadata requests; full download is explicitly restricted to Colab."""
from datetime import datetime, timezone
from pathlib import Path
import requests

DATASET = "mateuszbuda/lgg-mri-segmentation"
VERSION = 2
METADATA_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}"


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
