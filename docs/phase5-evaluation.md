# Original locked held-out evaluation procedure

I completed the final evaluation once on the frozen held-out cohort. The
[final notebook](../notebooks/NeuroGuard_Final_Submission.ipynb) reads its saved
aggregates. The procedure and approval gates below are the original experimental
record, not instructions to repeat the completed evaluation. No retraining,
threshold search or reassignment is part of the final readout or application.

## Frozen inputs

- `MyDrive/NeuroGuard/dataset-v2-seed42/split_manifest.json`, canonical identity
  `7514044281736b609e9d1c09c342a3f2514d0a2ca4dea9c4ce0d466cd86fe493`.
- BCE: `MyDrive/NeuroGuard/experiments/phase3-bce/bce-seed42-01/`, original source
  `454501448551cd0f05f4b74f7f77f9a67b9a0aab`, 25 completed/best epoch 18.
- Tversky: `MyDrive/NeuroGuard/experiments/phase4-tversky/tversky-seed42-01/`, source
  `1e6e6b6067d98641967c3dbaf4c49b605d1e81c5`, 30 completed/best epoch 23.
- Each run must contain `identity.json`, `config.json`, `best.pt`,
  `best_checksum.json`, `history.json`, `validation_reports.json`, `review_summary.json`.
  Missing or mismatched artifacts stop execution; never create replacement
  metadata or hashes merely to pass a check. Colocated hashes detect corruption,
  not authenticity: confirm these are your original trusted private runs.
- 17 held-out patients, 589 slices, 230 positive/359 empty; exact eligible patient
  counts come from the verified manifest and are checked against inference rows.
- Always-empty baseline; BCE/Tversky at .5; BCE selected .8; Tversky selected .5,
  explicitly a constraint-failure fallback. The Tversky .5 results are duplicated
  labels for the same predictions, not independent experiments.

The original [metric protocol](evaluation-protocol.md), models, preprocessing and
training configs remain fixed. Source verification compares implementation files
to **both** original training commits. Strict checkpoint shape loading, embedded
identity, checksum, best-history epoch and runtime/dependency checks occur before
test inference. Evaluation uses `eval()` and `torch.inference_mode()`, float32,
no augmentation or postprocessing. Only test file bytes are checked in preflight;
the full dataset audit and training-area quantiles are not repeated.

## Prespecified analysis

`configs/final_evaluation.json` freezes the five operating-point labels, 1,000
paired patient bootstrap replicates/seed42/95% percentile intervals, guardrails
and example policy. Resample all 17 patients with replacement, retaining repeated
patient weights; restrict each metric to eligible positive/size-group patients
within each draw. Empty-only draws are undefined and counted. Dice/recall
differences include positive and small/medium/large groups, for both comparisons.
Intervals are exploratory, without multiple-comparison correction or a significance
claim. They do not capture training-seed variation or external generalisation.

Report patient-macro positive Dice/recall/precision/IoU and size groups with patient
and slice denominators; empty-FP slice rate and mean foreground fraction; pooled
all-slice counts/metrics separately. All five approaches share the same denominator
checks. CSVs and a bar figure include the trivial baseline. Differences report pp
and relative percent separately (relative change NA when BCE is zero).

Before inference: stable order patient/numeric slice/image; choose the lower median
by annotation area in each nonempty size group and the first empty slice. After
inference: add each model's lowest-Dice positive and empty slice with most FP pixels
at .5, with sorted-order tie-breaks, deduplicated. Every figure shows the same FLAIR
channel 1 slice and GT alongside both arms at both operating points. Outcome-selected
failures are labelled; they do not estimate prevalence. If no empty FP exists, no
false-positive case is invented. Missing groups remain absent/NA.

## Exact Colab sequence

1. Review local code/tests. Source must be committed and pushed **only after your
   separate explicit approval**. No commit or push occurs as part of preparation.
2. Open `https://colab.research.google.com/github/hrb31/NeuroGuard/blob/COMMIT_SHA/notebooks/04_final_evaluation.ipynb`
   with the approved full 40-character SHA. Use a fresh browser Colab T4 runtime.
3. Run sections 1–4: environment, SHA input, token-free public clone and dependency
   installation/synthetic suite/pip check. Keep existing CUDA torch. If packages
   already imported change, restart and repeat at the same SHA.
4. Sections 5–7 mount Drive, verify the original manifest, accept dataset/upstream
   terms, reuse/download version2 **inside Colab only**, and resolve its canonical
   root. No Drive MCP/API or private GitHub authentication.
5. Section 8: explicitly confirm original trusted checkpoint directories. Run
   preflight; preserve its private report, report digest, pytest and environment
   outputs. Inspect the printed identities, hashes, configurations, epochs, cutoffs,
   denominators and selected/fallback labels. **Stop here for review.**
6. Section 9 is the final review gate. Set `PREFLIGHT_REVIEWED = True` and copy
   the printed digest into `REVIEWED_PREFLIGHT_SHA256` only after review. Type
   `EVALUATE FROZEN TEST ONCE` when the next cell prompts. This is your explicit
   approval to access test predictions, separate from source publication approval.
7. Section 10 revalidates source/runtime/data/checkpoint evidence and reloads the
   original weights. It reserves a single evaluation, runs inference and saves
   reports. Section 11 displays the concise aggregate summary and private figures.
   Do not adjust the protocol based on the outcomes.

Any identity/runtime/selection mismatch stops before test predictions: return the
error for review. Do not loosen checks. A `.pt` incompatible with safe
`weights_only=True` loading also stops; do not switch to unsafe pickle loading.
The source SHA must contain this implementation; an older remote SHA cannot run it.

## Persistence and recovery

Private outputs: `MyDrive/NeuroGuard/experiments/phase5-final-evaluation/`.
Preflight reports are timestamped below `preflight/`. A canonical
`EVALUATION_STARTED.json` prevents accidental repeats across run IDs. Results go
to `final-test-seed42-01/`. All earlier phases stay read-only.

The gate checks this workflow's result root for prior markers/results; it cannot
prove that no manual evaluation occurred elsewhere. Confirm that independently.
If preflight alone was interrupted, repeat sections 1–8 at the same source SHA;
no test predictions have been generated. If evaluation started, **preserve all
files and request recovery review**. Do not delete the marker, rename the run or
automatically rerun inference. Partial results may already have exposed the test.
The marker is an accidental-rerun safeguard, not a distributed transaction lock;
use one Colab runtime, not concurrent Drive clients.

Saved files: `final_test_metrics.json`, `comparison.json`, positive/size,
empty/pooled and difference CSVs, `paired_patient_bootstrap.json`,
`aggregate_comparison.png`, `metric_definitions.md`, `evaluation_provenance.json`,
private per-slice metrics and example-selection/FLAIR figures,
`artifact_hashes.json`, and concise `review_summary.json`. Failures write
`evaluation_failure.json`. All patient-level outputs and MRI figures stay private;
public aggregates/images require individual licensing/privacy review.

The retained analysis bundle includes `review_summary.json`, the three CSVs, bootstrap report,
aggregate comparison figure, preflight digest/identity-check output, pytest/pip
check output and any errors. Review private overlays locally and describe misses,
boundary errors and false positives; share only individually approved images.
Never send credentials, weights, patient identifiers or per-slice records publicly.

Interpret recall improvement and FP guardrails separately. Recall gains plus a
failed FP guardrail mean **partial support** only; neither model is clinically
validated. Explicitly compare actual test findings with validation, including
reversals. The completed results, application and write-up are linked from
the [README](../README.md).
