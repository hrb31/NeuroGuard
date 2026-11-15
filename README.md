# NeuroGuard

AI-assisted brain MRI abnormality segmentation prototype for the Veritas AI
International High School Hackathon 2026, **AI + Health** track.

**Research prototype only. Not a clinical diagnostic tool.** Predictions will
highlight model-flagged regions; they cannot diagnose a neurological condition,
rule out disease, or guide medical decisions.

This project is a prototype and is not medical advice.

## Status

Phase 2 audit/splitting/preprocessing code and a browser-Colab notebook are ready
for cloud execution. Release metadata is verified; actual dataset contents and
the final split are still pending execution and review. No model has been
implemented or trained, and no evaluation results exist.
The Gradio entry point is a placeholder; there is no working demo yet.

The dataset-root resolver verifies repeated directory copies using relative TIFF
filenames and file hashes before selecting one canonical root. Conflicting copies
stop the audit; raw data is never deleted or silently deduplicated. Real unique
counts and example images still need review in Colab sections 7-8 before freezing
the split. See [the workflow](docs/colab-workflow.md).

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
from 110 patients are expected, **pending cloud verification** of counts,
channels and image-mask correspondence. Kaggle's public metadata checked on
2026-11-15 reports version 2 and CC BY-NC-SA 4.0; acquisition is pinned to version 2.
Read the release and upstream terms before download or redistribution. See the
[dataset author's description](https://mateuszbuda.com/2017/12/01/brainseg.html).
Do not substitute a classification dataset or treat annotations as diagnoses.

Assign whole patients to training, validation and test sets with a fixed seed;
never split individual slices independently. Tune thresholds and select models
on validation data only. Lock the test set until the protocol is frozen.

Planned evaluation: Dice (primary), recall, precision, optional IoU, false-positive
rate on empty masks, size-stratified performance, overlays and failure cases.
The proposed aggregation, empty-mask conventions and size thresholds are recorded
in [the evaluation protocol](docs/evaluation-protocol.md) and must be frozen before training. Report positive
and empty slices separately so empty slices cannot inflate the primary score.

## Repository

- `notebooks/01_data_exploration.ipynb`: numbered Colab bootstrap/audit/split workflow.
- `src/`: acquisition, audit, patient splits, paired preprocessing and previews.
- `tests/`: synthetic integrity, patient separation, preprocessing and notebook checks.
- `docs/`: Colab reproduction steps and evaluation methodology.
- `results/`: curated evaluation summaries and figures after experiments.
- `app.py`: future Gradio entry point.
- `requirements.txt`: dependency template; not a tested environment lock.
- `requirements-audit.txt`: Phase 2 dependencies only; preserves Colab's torch.

Private instructions and references live in Git-ignored `AGENTS.md` and
`local-context/`. Data, checkpoints, environments and secrets are excluded too.

## Development environment

The user has verified browser Colab CUDA/T4 access. Use that working runtime;
do not spend time repairing the VS Code Colab extension. See
[the exact VS Code -> GitHub -> Colab steps](docs/colab-workflow.md).

Locally the existing `.venv` uses Python 3.14.2 and runs the Phase 2 synthetic
tests. This does not validate the separate Colab environment. For local audit work:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-audit.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Select `.venv\Scripts\python.exe` in VS Code. Training is intended for Google
Colab GPU. Check the [official PyTorch installer](https://pytorch.org/get-started/locally/)
for runtime-specific wheels; do not replace Colab's working GPU build blindly.
Record Python, package, CUDA and device versions in each experiment, and freeze
the first working environment before the matched comparison. Current dependency
ranges are proposals, not a reproducibility guarantee.

## Next task

After approval to push this code, open the exploration notebook at the exact Git
commit in browser Colab and run numbered sections 1-8. Review real outputs, fill
the section-9 review gate, and run sections 9-10 to freeze/persist the split.
Return outputs for verification before starting any model work. Raw MRI data
downloads only inside Colab, never to this Windows workspace.

## Attribution and limitations

Cite the dataset release and upstream sources according to their verified terms
before redistributing any examples or derivatives. The dataset is listed as
CC BY-NC-SA 4.0; this repository does not grant independent rights to data or
derived artifacts. No code license has been selected.
U-Net: [Ronneberger et al.](https://arxiv.org/abs/1505.04597).
Tversky loss: [Salehi et al.](https://arxiv.org/abs/1706.05721).

Expected limitations include a small patient cohort, correlated 2D slices,
dataset-specific annotations and acquisition conditions, and no clinical or
external validation. Report negative results and subgroup counts honestly.
Before submission, document human contributions, external code and AI assistance.
Phase 2 code/notebook/docs were prepared with OpenAI Codex assistance. The team
must review, understand and verify the cloud results before submission.
The Health brief also values demographic subgroup checks; assess available public
metadata and report when sample sizes or missing attributes prevent these checks.
