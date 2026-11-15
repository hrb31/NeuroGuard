# NeuroGuard

Brain MRI abnormality segmentation research prototype for the Veritas AI
International High School Hackathon 2026, **AI + Health** track.

**This project is a prototype and is not medical advice.** It cannot diagnose a
condition, rule out disease or guide medical decisions.

## Research question

Does replacing binary cross-entropy with Tversky loss improve U-Net segmentation
recall, particularly for small annotated regions, without unacceptable false
positives? Improvement is a hypothesis, not a finding.

Three comparison levels: an always-empty mask (Model 0), a small U-Net trained
with `BCEWithLogitsLoss` (Model 1), and the identical U-Net with Tversky loss
(`alpha=.3` FP, `beta=.7` FN; Model 2, Phase 4 implementation prepared). The experiment
changes the loss while preserving data, preprocessing, architecture, seed policy,
optimisation budget and selection rules.

## Current status

Phase 2 is complete on actual Colab evidence verified by the user and supplied
in the Phase 3 request: **110 patients, 3,929 image/mask pairs**, 1,373 positive
and 2,556 empty masks, dimensions 256×256×3, zero integrity errors and zero
cross-patient duplicate groups. Representative small/large/empty overlays and
preprocessing were reviewed; this was not exhaustive manual annotation validation.

The release README identifies stored channels as pre-contrast (0), FLAIR (1) and
post-contrast (2); missing sequences may substitute FLAIR. Masks annotate
FLAIR-associated abnormalities. These are stored MRI sequences, not RGB colours.

The immutable seed42 patient split is saved privately in Drive:

| Subset | Patients | Slices | Positive-mask slices |
|---|---:|---:|---:|
| Training | 77 | 2,719 | 934 |
| Validation | 16 | 621 | 209 |
| Test | 17 | 589 | 230 |

**Phase 3 is complete on user-verified Colab validation evidence:** 25 epochs,
best epoch 18. The always-empty baseline has Dice/recall 0, empty FP 0% and pooled
pixel accuracy 99.05%. BCE validation findings:

| Metric | BCE t=.5 | BCE selected t=.8 |
|---|---:|---:|
| Positive-slice patient-macro Dice | 0.73794 | 0.69258 |
| Recall | 0.81314 | 0.66549 |
| Precision | 0.73336 | 0.81206 |
| Empty-mask FP rate | 19.175% | 9.466% |
| Small-region Dice | 0.62082 | 0.56990 |
| Small-region recall | 0.76215 | 0.53931 |

Threshold .8 satisfies the frozen <=10% empty-mask FP constraint, with lower
recall, including on small annotations. Curves/overlays were reviewed by the user.
See [verified aggregate evidence and limitations](docs/phase3-validation.md).
These are validation findings, not final test results.

**Phase 4 implementation is prepared; actual Tversky GPU training is pending.**
It reuses the identical U-Net and shared training/resume code, changing loss alone.
No final test predictions or working Gradio demo exists.

## Data and evaluation

[Mateusz Buda's LGG MRI segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation),
version 2, is listed as CC BY-NC-SA 4.0 with TCIA/TCGA provenance. Review release
and upstream terms before downloading or redistributing derivatives. Data
downloads happen only in browser Colab, never in the Windows source workspace.
The existing canonical-root resolver compares all TIFF filenames and hashes
before selecting between identical directory copies; conflicting copies stop.

Never regenerate the frozen assignments. Patient-disjoint validation selects
checkpoints and thresholds; test evaluation stays locked until Phase 5. Primary
Dice averages positive slices within each eligible patient, then patients equally.
Report recall/precision, empty-mask false positives, training-derived size groups
and denominators separately. Background-heavy pixel accuracy can hide complete
failure to detect foreground. See [the frozen protocol](docs/evaluation-protocol.md).

## Run Phase 4

Use [03_train_tversky.ipynb](notebooks/03_train_tversky.ipynb) in browser Colab
with a T4. The [Phase 4 guide](docs/phase4-training.md) gives exact inputs, numbered
execution steps, the smoke-review gate, resume instructions and artifacts to return.
The repository is public; no GitHub token is needed. Do not rerun completed BCE.
Source must be available at an approved
full Git commit; opening the notebook alone does not clone its modules. No push
or publication occurs automatically.

The [frozen BCE configuration](configs/bce.json) uses a 3→1 U-Net with four
downsamplings (16/32/64/128/256), **1,942,577 trainable parameters**, Adam .001,
batch16, max30 epochs, patience7 and
seed42. Outputs are logits. Paired preprocessing and training augmentation reuse
Phase 2. Private Drive checkpoints include optimiser/RNG state, configuration,
manifest identity, code SHA, environment, histories and best validation weights.
The [Tversky configuration](configs/tversky.json) changes loss and experiment
identity only. Its per-slice mean includes empty masks, with smoothing `1e-6` in
numerator and denominator. Empty-mask gradients can be extremely small; the
Phase 4 guide documents this limitation before the definitive run.

Local synthetic checks (install the appropriate CPU/GPU torch build separately):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

`requirements-training.txt` deliberately preserves Colab's installed CUDA torch.
Record the actual cloud environment for the matched experiment. Dependency ranges
are not a tested cross-platform lock. Audit-only tests also work without torch;
training tests then skip explicitly.

## Repository and remaining roadmap

- `notebooks/01_data_exploration.ipynb`: original Phase 2 audit/evidence workflow.
- `notebooks/02_train_bce.ipynb`: Phase 3 validation baseline and BCE experiment.
- `notebooks/03_train_tversky.ipynb`: matched Phase 4 training and validation comparison.
- `src/`: data integrity, preprocessing, model, metrics, loaders, training and plots.
- `configs/bce.json`, `configs/tversky.json`, `tests/`, `docs/`: frozen settings, checks and reproduction.
- `results/`: only future reviewed summaries; `app.py`: future Gradio placeholder.

Phases 1–3 complete on user-verified evidence; Phase 4 prepared, cloud run pending;
Phase 5 locked final
three-model evaluation/error analysis; Phase 6 Gradio/demo/submission. A mandatory
final refinement and scientific/submission audit follows the core work, including
a separate polished judge-facing notebook with genuine reviewed outputs. It is
not performed during Phase 4.

Data, patient identifiers, checkpoints, credentials, environments, private context
and unreviewed notebook outputs stay out of Git. Do not publish derivative images
or weights without licensing/privacy review. No code licence has been selected.

## Sources and limitations

Dataset: [release](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
and [author's description](https://mateuszbuda.com/2017/12/01/brainseg.html).
Architecture: [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597).
Loss: [Salehi et al., Tversky](https://arxiv.org/abs/1706.05721).
Implementation guidance: [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html),
[BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
and [checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Code, tests, notebooks and documentation were prepared with **OpenAI Codex**
assistance. The student reviews the design, executes and validates experiments,
interprets findings and must understand and explain the methods. Do not represent
AI-generated code as entirely manually authored.

Limits include a small cohort, correlated 2D slices, dataset-specific annotations,
sequence substitution, possible near-duplicates not caught by exact hashes and
no external/clinical validation. Size strata are annotation burden, not 3D lesion
volume or demographic fairness. Assess public demographic metadata and report
when missing attributes or inadequate counts prevent meaningful subgroup checks.
Report misses, false alarms and negative results honestly.
