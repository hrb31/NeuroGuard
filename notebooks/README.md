# Notebooks

Use the existing `01_data_exploration.ipynb` as the single Phase 2 notebook. It
bootstraps a pinned Git commit in browser Colab, audits version-2 data, previews
training-only examples and freezes patient-disjoint splits after a review gate.
See ../docs/colab-workflow.md for numbered execution/manual authentication steps.
Later: matched BCE and Tversky runs, then locked evaluation. No training yet.

Keep reusable logic in `src/`. Do not commit private paths, tokens, patient
identifiers or unreviewed image outputs embedded in notebooks. Clear sensitive
outputs before staging. Record configuration and environment with every run.
