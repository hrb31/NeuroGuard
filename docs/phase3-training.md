# Phase 3 execution and evidence

**Completed on user-verified Colab validation evidence supplied 2026-11-15.**
See [verified findings](phase3-validation.md). Preserve the original BCE artifacts;
do not rerun this notebook for Phase 4. Use [the Tversky guide](phase4-training.md).
The execution instructions below document the completed Phase 3 workflow.

Phase 2 counts/channel review and split completion are supplied as verified by
the user in the Phase 3 request. This local implementation does not rerun that
cloud audit. Never replace the saved assignments or original evidence.

## Before opening Colab

After reviewing the public diff and explicitly approving a push, obtain the full
40-character commit with `git rev-parse HEAD`. Open:

`https://colab.research.google.com/github/hrb31/NeuroGuard/blob/COMMIT_SHA/notebooks/02_train_bce.ipynb`

Replace COMMIT_SHA with that exact commit. A local uploaded notebook cannot fetch
unpublished source. The repository is public and requires no GitHub token. Do not download MRI data locally or upload private patient data.

Have the **original Phase 2 Frozen manifest SHA256** ready (section 9 output or
the original evidence archive). It is the canonical JSON identity stored under
`manifest_sha256`, not the file-byte digest. The notebook requires it as an
independent identity check; do not paste an arbitrary newly generated value.

## Numbered notebook execution

1. Select a T4 GPU and run the Colab/CUDA check.
2. Enter the reviewed full commit SHA.
3. Clone and verify clean source at that SHA.
4. Install supporting dependencies, run synthetic tests and pip check. Keep CUDA torch.
5. Mount Drive; choose `bce-seed42-01` (or another simple new ID) and NEW. Use
   the same ID with RESUME after interruption. Review the printed config.
6. Enter the original Phase 2 hash. Verify frozen counts and load size cutoffs.
7. Review terms and type ACCEPT; obtain pinned version2 in Colab. Canonical-root
   comparison and hashes verify all saved pairs. No new split or area calibration.
8. Check seeded train/validation loaders, binary masks, U-Net shape/parameter count.
9. Compute the always-empty validation baseline and eligible counts.
10. Run the 100-step four-training-slice debugging check, inspect its loss, Dice
    and overlay, and set `SMOKE_REVIEWED=True` only if satisfied. Loss decrease is
    a minimum check, not proof that foreground is fitted. Return failures for review.
11. Type TRAIN to begin/resume the frozen BCE fit. A fresh seeded model is built;
    smoke weights/optimiser are discarded. Do not silently alter batch or budget.
12. Evaluate the selected best model on validation at .5 and all declared search
    thresholds; save reports, curves and representative FLAIR predictions.
13. Save run records/checksums and verify the original manifest file is unchanged.
14. Return the review summary and review figures. Stop before Phase 4.

If the T4 cannot fit batch16, stop and record the OOM. The proposed fallback is
batch8 for both definitive arms, but this material protocol change requires a
recorded decision/approval before running a new definitive experiment. Do not
claim an aborted run as a completed experiment.

## Frozen configuration

`configs/bce.json` is the common preparation for the matched comparison:
three inputs/one raw-logit output, four downsamplings, widths16/32/64/128/256;
two conv+BatchNorm+ReLU layers per level, 1,942,577 trainable parameters,
transposed-convolution decoder/skip
concatenation, no pretrained weights. Odd spatial shapes align by interpolation;
the experiment uses native 256x256. No sigmoid is applied before BCEWithLogitsLoss.

Adam lr=.001, weight_decay=0 (standard betas .9/.999 and eps1e-8), batch16,
float32, seed42, max30 epochs, patience7. No scheduler, class weighting, AMP,
sampling rebalance or postprocessing. Initialise the future identical architecture
from the same seed before constructing its optimiser; preserve all other settings.

Save every strictly higher validation positive-slice patient-macro Dice at .5;
earliest epoch wins ties. The existing min_delta1e-4 governs patience resets,
separately from best-checkpoint selection. After selection run the one frozen
validation threshold search; see evaluation-protocol.md. Guardrails of 10% absolute
empty-mask FP and 5pp relative increase are research choices, not clinical safety.

Reuse Phase 2 centred square padding, bilinear image/nearest mask resize,
per-image/channel nonzero z-score, constant→zero; joint horizontal flip p=.5 and
rotation±10° p=.5 only in training. No fitted cohort statistics. Per-sample/epoch
augmentation RNG and seeded worker/shuffle generators make worker scheduling
irrelevant to sampled transforms. Deterministic torch algorithms are required;
unsupported kernels stop rather than silently disabling the constraint.

## Drive outputs and resume

Original Phase 2 evidence: `MyDrive/NeuroGuard/dataset-v2-seed42/` (read-only).
New experiment: `MyDrive/NeuroGuard/experiments/phase3-bce/RUN_ID/`.
Timestamped smoke/preflight evidence: sibling `preflight/RUN_ID/TIMESTAMP/`.

The run holds `identity.json`, `config.json`, `environment.txt`, `history.json`,
two rotating state checkpoints, `latest.json`/`previous.json`, `best.pt` and its
checksum, `validation_reports.json`, `always_empty_validation.json`,
`smoke_debug_only.json`, `training_curves.png`, `figures/`, `review_summary.json`
and `artifact_hashes.json`. The identity includes Python/packages/CUDA/GPU,
configuration, model parameters, code SHA, dataset fingerprint, canonical manifest
hash and file-byte hash, channels and original size cutoffs.

Each state includes current and best model, Adam state, completed epoch, history,
patience state and Python/NumPy/torch/CUDA RNG. The next inactive slot is saved
and hashed before an atomic latest-pointer replacement; the last completed epoch
stays available if a write is interrupted. The selected best state is in the
training checkpoint even before final best.pt export. Keep the whole run directory.

After a disconnect, use the SAME notebook SHA, run ID and environment, rerun 1–10,
choose RESUME, then 11–14. Resume restores the last completed epoch; an unfinished
epoch is replayed. The total budget stays 30 and an early-stopped run does not
restart. History/checkpoint and report exports can be reconstructed by resuming
a completed run. If a disconnect precedes the first completed checkpoint, a
matching saved run identity permits explicit RESUME to replay epoch 1 from the
original seed and budget; no new run or extra completed epochs are granted.
Bitwise reproducibility is verified locally for an interrupted
CPU synthetic run, not promised across changed packages/hardware or Colab resets.
PyTorch documents [cross-platform/version reproducibility limits](https://docs.pytorch.org/docs/stable/notes/randomness.html).

A checksum/identity mismatch stops. Preserve evidence and investigate; never edit
the split, identity or checkpoint to bypass it. previous.json identifies the
previous completed slot for deliberate reviewed recovery if necessary; the code
does not silently roll back. Load only your own trusted saved checkpoint: full
training state uses Python pickle (`weights_only=False`). No manual individual
checkpoint management is needed for normal resume.

## Return for review

Copy section 14's `review_summary.json`, pytest output, smoke result/overlay,
training curve and representative validation overlays. It includes baseline and
BCE patient-macro Dice/recall/precision/IoU, empty FP/fraction, size-group patient
and slice counts, supplementary pooled metrics, completed/best epochs, default
and selected thresholds, code and manifest identities, and observed training time.
Report errors and interruptions. Do not send tokens, raw TIFFs, patient-level
records, manifest or full weights. Pixel accuracy is supplementary: a predictor
can score highly by returning background everywhere while detecting nothing.

Phase 3 completion rests on genuine user-verified Colab outputs, not local
synthetic tests. Final three-model test evaluation, Gradio and mandatory
submission refinement remain later phases.
