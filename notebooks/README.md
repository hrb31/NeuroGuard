# Notebooks

Use the existing `01_data_exploration.ipynb` as the single Phase 2 notebook. It
bootstraps a pinned Git commit in browser Colab, audits version-2 data, previews
training-only examples and freezes patient-disjoint splits after a review gate.
See ../docs/colab-workflow.md for numbered execution/manual authentication steps.
Phase 2 is complete on actual Colab evidence verified/supplied by the user.
Never regenerate its frozen assignments. Use `02_train_bce.ipynb` for Phase 3:
numbered source bootstrap, original manifest identity, canonical data preflight,
always-empty validation baseline, train-only smoke review, BCE fit/resume and
validation plots/reports. See ../docs/phase3-training.md for exact inputs/steps.
Phase 3 cloud execution is complete on user-verified validation evidence.
Use `03_train_tversky.ipynb` for Phase 4: public pinned clone, original manifest
and BCE identity checks, loss diagnostics, training-only smoke gate, fresh Tversky
fit/resume, validation threshold search and aggregate BCE comparison. See
../docs/phase4-training.md. Phase 4 is complete on user-verified validation evidence.
Use `04_final_evaluation.ipynb` for Phase 5: original checkpoint/manifest preflight,
explicit review gate, fixed thresholds and patient-bootstrap/FLAIR comparisons.
Run sections 1–8, STOP for review, then explicitly approve sections 9–10.
See ../docs/phase5-evaluation.md. No actual test results exist from local preparation. A separate polished judge-facing notebook is a mandatory
final refinement milestone after modelling, evaluation and Gradio, not now.

Keep reusable logic in `src/`. Do not commit private paths, tokens, patient
identifiers or unreviewed image outputs embedded in notebooks. Clear sensitive
outputs before staging. Record configuration and environment with every run.
