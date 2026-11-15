# Planned automated checks

Run `python -m pytest -q` from repository root. test_data.py generates synthetic
TIFF pairs and checks missing/corrupt/orphan masks, labels and dimensions, duplicate
detection, disjoint/exhaustive/reproducible patient splits, immutable manifests,
paired transforms, finite empty/constant inputs, preview files and notebook syntax.
These checks do not constitute real dataset, remote Colab or model validation.
Directory-layout cases cover single roots, identical nested wrapper copies,
conflicting/missing/extra/renamed image or mask files, unassigned TIFFs, and
canonical-root audit/preview/freeze integration without weakening duplicate checks.
Future phases add metric edge cases, U-Net tensor shapes and loss checks.
