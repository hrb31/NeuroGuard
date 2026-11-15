# NeuroGuard

AI-assisted brain MRI abnormality segmentation prototype for the Veritas AI
International High School Hackathon 2026, **AI + Health** track.

**Research prototype only. Not a clinical diagnostic tool.** Predictions will
highlight model-flagged regions; they cannot diagnose a neurological condition,
rule out disease, or guide medical decisions.

This project is a prototype and is not medical advice.

## Status

Initial repository scaffold only. No dataset has been downloaded or verified,
no model has been implemented or trained, and no evaluation results exist.
The Gradio entry point is a placeholder; there is no working demo yet.

## Research question

Does replacing binary cross-entropy with Tversky loss improve U-Net recall,
particularly for small annotated abnormal regions, without an unacceptable
increase in false positives?

The main comparison will use the **same small PyTorch U-Net**, with
`BCEWithLogitsLoss` versus Tversky loss (`alpha=0.3` for false positives,
`beta=0.7` for false negatives). Splits, preprocessing, architecture and training
budget will be matched. An always-empty mask will provide a non-learning baseline.
Improvement is a hypothesis, not a finding.

## Data and evaluation plan

Use [Mateusz Buda's LGG MRI segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation),
with MRI images and annotated FLAIR abnormality masks. Approximately 3,929 slices
from 110 patients are expected, **pending local verification** of counts,
channels, image-mask correspondence and licensing. See the
[dataset author's description](https://mateuszbuda.com/2017/12/01/brainseg.html).
Do not substitute a classification dataset or treat annotations as diagnoses.

Assign whole patients to training, validation and test sets with a fixed seed;
never split individual slices independently. Tune thresholds and select models
on validation data only. Lock the test set until the protocol is frozen.

Planned evaluation: Dice (primary), recall, precision, optional IoU, false-positive
rate on empty masks, size-stratified performance, overlays and failure cases.
The proposed aggregation, empty-mask conventions and size thresholds are recorded
in the local specification and must be frozen before training. Report positive
and empty slices separately so empty slices cannot inflate the primary score.

## Repository

- `notebooks/`: data exploration and Colab experiment notebooks (not yet authored).
- `src/`: reusable data, model, loss, training and evaluation code (not yet implemented).
- `tests/`: future lightweight checks for pairing, splits, metrics and tensor shapes.
- `results/`: curated evaluation summaries and figures after experiments.
- `app.py`: future Gradio entry point.
- `requirements.txt`: dependency template; not a tested environment lock.

Private instructions and references live in Git-ignored `AGENTS.md` and
`local-context/`. Data, checkpoints, environments and secrets are excluded too.

## Development environment

Recommended starting interpreter: **Python 3.11**. A Python runtime was not found
on PATH during setup, so a virtual environment and dependency installation have
not been validated. After installing Python, from this folder in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Select `.venv\Scripts\python.exe` in VS Code. Training is intended for Google
Colab GPU. Check the [official PyTorch installer](https://pytorch.org/get-started/locally/)
for runtime-specific wheels; do not replace Colab's working GPU build blindly.
Record Python, package, CUDA and device versions in each experiment, and freeze
the first working environment before the matched comparison. Current dependency
ranges are proposals, not a reproducibility guarantee.

## Next task

Acquire the dataset privately after verifying access and usage terms; inventory
patient IDs, images and masks; inspect alignment and channels; then save and test
a reproducible patient-disjoint split manifest. Do not start training yet.

## Attribution and limitations

Cite the dataset release and upstream sources according to their verified terms
before redistributing any examples or derivatives. Dataset licensing is unresolved;
this repository does not grant rights to the data. No code license has been selected.
U-Net: [Ronneberger et al.](https://arxiv.org/abs/1505.04597).
Tversky loss: [Salehi et al.](https://arxiv.org/abs/1706.05721).

Expected limitations include a small patient cohort, correlated 2D slices,
dataset-specific annotations and acquisition conditions, and no clinical or
external validation. Report negative results and subgroup counts honestly.
Before submission, document human contributions, external code and AI assistance.
The Health brief also values demographic subgroup checks; assess available public
metadata and report when sample sizes or missing attributes prevent these checks.
