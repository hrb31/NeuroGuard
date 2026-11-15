# Planned automated checks

Run `python -m pytest -q` from repository root. test_data.py generates synthetic
TIFF pairs and checks missing/corrupt/orphan masks, labels and dimensions, duplicate
detection, disjoint/exhaustive/reproducible patient splits, immutable manifests,
paired transforms, finite empty/constant inputs, preview files and notebook syntax.
These checks do not constitute real dataset, remote Colab or model validation.
Directory-layout cases cover single roots, identical nested wrapper copies,
conflicting/missing/extra/renamed image or mask files, unassigned TIFFs, and
canonical-root audit/preview/freeze integration without weakening duplicate checks.
Phase 3 adds test_metrics.py (edge cases, two-stage patient aggregation, frozen
size groups and threshold constraints) and test_training.py (manifest identity,
read-only verification, no-test loaders, seeded workers, U-Net shapes/backward,
finite BCE gradients, validation baseline, train-only tiny loss decrease and
checkpoint reload, interrupted-vs-uninterrupted exact CPU resume, checksum/config
rejection and clean training-notebook syntax). CUDA test skips on CPU; all training
tests skip explicitly if torch is absent. Local synthetic checks are not real
validation results or proof of T4 memory capacity.

conftest.py selects the noninteractive Agg plotting backend for tests, avoiding
Windows Tk dependencies and GUI windows while verifying saved figures.

Phase 4 adds test_losses.py (analytic soft counts/FP-FN order, native-grid empty,
tiny/large/background/perfect/mixed masks, signed gradients, numerical gradcheck,
empty-mask gradient/rounding limitations, matched configs/initialisation/parameter
count and CUDA backward when available). Shared resume and tiny-overfit tests run
both losses; test_comparison.py checks percentage-point units, missing metrics and
cohort mismatch. Both training notebooks must compile with no outputs. These are
synthetic engineering checks, not Tversky MRI findings or a cloud execution claim.
