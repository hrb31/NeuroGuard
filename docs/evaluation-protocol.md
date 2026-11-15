# Frozen modelling and evaluation protocol — Phase 3

Frozen on 2026-11-15 before full BCE training, using the defaults authorised in
the Phase 3 request. Phase 2 is complete on verified Colab evidence supplied by
the user. Actual baseline/BCE validation findings remain pending cloud execution.
No final test predictions have been accessed. See configs/bce.json and
docs/phase3-training.md for training implementation and provenance.

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

Checkpoint selection: best validation primary Dice at threshold .5,
same budget/early-stopping rule for BCE and Tversky. After checkpoint selection,
search thresholds .10,.15,...,.90 using validation only. Maximise primary Dice
subject to empty-mask FP rate <=10%; tie-break lower FP rate, then higher threshold.
If no threshold qualifies, retain .5 and report constraint failure. If validation
has no empty masks, mark the constraint unevaluable and use Dice/high-threshold
tie-break. Also report both models at .5 for an unconfounded comparison.

Frozen acceptance guardrail: absolute empty-mask FP rate <=10% and Tversky
increase over BCE <=5 percentage points. These are research tolerances, not
clinically validated thresholds. These are fixed before training, never after
seeing test outcomes. Report recall/Dice tradeoffs and negative results when unmet.

Store every strictly better checkpoint; earliest epoch wins exact ties. Patience7
uses min_delta=1e-4 relative to the last patience-reset reference, independently
of saving the actual best checkpoint. Both arms use this rule and max30 epochs.

Split audit uses all samples only for integrity and counts. Preview only training
slices. Do not inspect test predictions or tune transforms, area boundaries,
loss/architecture, checkpoint or thresholds on the test set. At final evaluation,
lock both checkpoints, thresholds and configuration before opening test predictions.

## Frozen data and preprocessing basis

- User-verified release README: pre-contrast0, FLAIR1, post-contrast2; missing
  sequences may substitute FLAIR. Keep all three channels for both arms.
- User-verified integrity/representative preprocessing review: 3,929 pairs,
  110 patients, native256x256x3, no integrity errors/cross-patient duplicate groups.
- Existing seed42 assignments: train77/2,719/934 positive; validation16/621/209;
  test17/589/230. Never regenerate or change. Load original manifest hash and
  training-derived numeric area cutoffs; no validation/test recalibration.
- Existing src/preprocessing.py: centred square padding, image bilinear/mask
  nearest; per-image/channel nonzero z-score after geometry; train-only joint
  horizontal flip p=.5 and rotation +/-10deg p=.5. No evaluation augmentation.
- Small four-level U-Net16/32/64/128/256, Conv+BatchNorm+ReLU; Adam .001,
  weight_decay0, batch16, max30/patience7/seed42/float32. No pretrained encoder,
  class weighting, scheduler, postprocessing or AMP. Batch-size changes require
  review and matched definitive arms; no automatic OOM-driven change.
- Version2 release terms still apply. Public derivatives need attribution,
  licensing and privacy review separately from this private experiment.

Demographic fairness checks require suitable public metadata and adequate counts.
Do not infer demographics from MRI pixels or call size strata a fairness audit.
