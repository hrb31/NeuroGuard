# Verified Phase 3 validation findings

Genuine Colab results verified and supplied by the user in the Phase 4 request
on 2026-11-15. These are validation findings, not final test results, and were
not recalculated by the agent. Cohort: 16 patients, 621 slices, 209 positive masks.
U-Net: 1,942,577 parameters; seed42, Adam .001, batch16, max30/patience7;
25 completed epochs, best checkpoint epoch18.

| Validation metric | Always empty | BCE t=.5 | BCE selected t=.8 |
|---|---:|---:|---:|
| Positive-slice patient-macro Dice | 0.00000 | 0.73794 | 0.69258 |
| Recall, same aggregation | 0.00000 | 0.81314 | 0.66549 |
| Precision, same aggregation | — | 0.73336 | 0.81206 |
| Empty-mask FP rate | 0% | 19.175% | 9.466% |
| Small-region Dice | — | 0.62082 | 0.56990 |
| Small-region recall | — | 0.76215 | 0.53931 |

Always-empty pooled pixel accuracy was 99.05% despite zero positive-slice
Dice/recall, illustrating background imbalance. The selected .8 threshold meets
the predeclared <=10% empty-mask FP constraint. Compared with .5, it lowers empty
FP and raises precision but loses recall and Dice, including on small regions.
The user visually reviewed training curves and segmentation overlays: the model
learns meaningful abnormality segmentation while missing some small annotations,
particularly at the higher threshold. This does not establish clinical usefulness.

These are rounded supplied aggregates. IoU, per-size eligible denominators,
medium/large scores, exact run time and artifact hashes were not supplied; do not
invent them. Full private Phase 3 reports stay unchanged. The machine-readable
[aggregate reference](bce-validation-reference.json) supports the upcoming
[Phase 4 comparison](phase4-training.md). No Tversky findings exist yet.
