# Evaluation protocol before experiments

No model predictions or performance results exist for Phase 2. This document
operationalises the existing project specification; guardrails remain proposals.

## Aggregation and empty masks

At a selected probability threshold, TP/FP/FN count foreground pixels on the
256x256 processed grid. Dice=2TP/(2TP+FP+FN), recall=TP/(TP+FN),
precision=TP/(TP+FP), optionally IoU=TP/(TP+FP+FN).

Primary: Dice over GT-positive slices, mean within each patient, then equal mean
over eligible patients. Recall and precision use the same positive-slice
patient-macro scheme. Report eligible patient and slice counts. A positive mask
with empty prediction scores zero for Dice, recall, precision and IoU.

Both masks empty: per-slice Dice/IoU=1; recall/precision undefined (NA). GT empty,
prediction nonempty: Dice/IoU/precision=0, recall NA. Empty slices never inflate
the primary positive-slice Dice. A cohort with no eligible patients is NA.

Empty-mask FP rate: share of empty-GT slices with any predicted foreground.
Also report their mean predicted foreground fraction and pooled FP/background
pixel rate. No empty slices means NA, not zero. Patients with empty slices are
not necessarily disease-free. Supplementary all-slice micro metrics pool counts;
undefined denominators stay NA, distinct from per-slice both-empty Dice=1.

## Size analysis and uncertainty

Use original-resolution positive mask area/image area. Compute q1/3 and q2/3 only
on training positives; save in frozen manifest. Small: a<=q1/3; medium:
q1/3<a<=q2/3; large: a>q2/3. Ties may produce empty groups; report NA and counts.
This measures slice annotation burden, not lesion count or 3D volume. Within each
group use the same patient-macro positive-slice aggregation. Optional uncertainty:
paired patient bootstrap, seed42, 1000 replicates, never independent slice bootstrap.

## Selection and test isolation

Proposed checkpoint selection: best validation primary Dice at threshold .5,
same budget/early-stopping rule for BCE and Tversky. After checkpoint selection,
search thresholds .10,.15,...,.90 using validation only. Maximise primary Dice
subject to empty-mask FP rate <=10%; tie-break lower FP rate, then higher threshold.
If no threshold qualifies, retain .5 and report constraint failure. If validation
has no empty masks, mark the constraint unevaluable and use Dice/high-threshold
tie-break. Also report both models at .5 for an unconfounded comparison.

Proposed acceptance guardrail: absolute empty-mask FP rate <=10% and Tversky
increase over BCE <=5 percentage points. These are research tolerances, not
clinically validated thresholds. Freeze before training, never after seeing test
outcomes. Report recall/Dice tradeoffs and negative results even when unmet.

Split audit uses all samples only for integrity and counts. Preview only training
slices. Do not inspect test predictions or tune transforms, area boundaries,
loss/architecture, checkpoint or thresholds on the test set. At final evaluation,
lock both checkpoints, thresholds and configuration before opening test predictions.

## Still to freeze before model development

- Actual channel meanings/input selection with primary-source evidence.
- Real audit and processed-image review, including survival of tiny masks on resize.
- Dataset fingerprint, assignments, group counts, duplicate-resolution decision
  and suitability of the seeded split. No performance-based re-splitting.
- Preprocessing configuration (currently all stored channels, 256x256 square
  padding/resize, per-image nonzero z-score, train-only paired flip/rotation).
- Optimiser/budget/architecture proposals and the FP tolerances above.
- Dataset terms and reviewed attribution for any publicly shown derivatives.

Demographic fairness checks require suitable public metadata and adequate counts.
Do not infer demographics from MRI pixels or call size strata a fairness audit.
