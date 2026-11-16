# Reviewed Phase 4 validation findings

I completed and reviewed this Tversky validation run in Colab on 2026-11-15.
These are validation findings; they are not final test results.
Cohort: 16 patients, 621 slices,
209 positive and 412 empty masks. Both arms use the same 1,942,577-parameter U-Net.
Tversky completed 30 epochs; best epoch 23. Frozen alpha=.3 FP, beta=.7 FN,
smooth=1e-6, per-slice batch mean including empty masks. BCE completed 25 epochs,
best epoch 18. Training-loss magnitudes are different objectives.

| Patient-macro positive/size metric unless labelled | BCE .5 | Tversky .5 / fallback | BCE selected .8 |
|---|---:|---:|---:|
| Dice | .73794 | .79822936 | .69257895 |
| Recall | .81314 | .88439719 | .66549250 |
| Precision | .73336 | .75929147 | .81206406 |
| Small-region Dice | .62082 | .68867639 | .56989910 |
| Small-region recall | .76215 | .82965727 | .53931411 |
| Empty-mask FP slice rate | .19175 | .73300971 | .09466019 |

At .5, Tversky minus rounded BCE: Dice +6.028936 percentage points, recall
+7.125719 pp, small Dice +6.785639 pp, small recall +6.750727 pp and empty FP
+54.125971 pp. These are percentage-point differences, not relative percentages;
BCE inputs are the rounded reference aggregates. No uncertainty from aggregates.

BCE selected .8 meets the validation <=10% empty-FP constraint. No tested Tversky
threshold met it: retain .5 with `fp_constraint_unmet_fallback`. The validation
evidence supports the recall component of the hypothesis, but fails the absolute
10% and relative +5 pp research FP guardrails. Thus the hypothesis is **only
partially supported on validation**. Neither guardrail is a clinical safety limit.

I reviewed a missed small annotation, successful medium-region prediction,
imperfect large boundaries and visible false positives on an empty-mask slice.
IoU, medium/large metrics, size denominators, timings and hashes are not
reported in this rounded summary. The [final notebook](../notebooks/NeuroGuard_Final_Submission.ipynb)
contains the completed held-out findings from saved aggregates.

Original private runs and manifest remain unchanged. See
[Phase 5 execution and review gate](phase5-evaluation.md). AI assistance and dataset
attribution remain as stated in the README; no external or clinical validation.
