# Phase 4: matched Tversky experiment

Phase 3 is complete on genuine validation evidence verified and supplied by the
user on 2026-11-15. The agent has not rerun BCE or inspected its private artifacts.
Phase 4 code is prepared locally; actual Tversky training/results remain pending
execution in the user's browser Colab account. Phase 5 remains locked.

## Frozen intervention

`configs/tversky.json` differs from `configs/bce.json` only in experiment identity
and loss. `src/model.py` is unchanged: three inputs, one logit, widths
16/32/64/128/256, four downsamplings, BatchNorm/ReLU, 1,942,577 parameters.
Seed42 initialisation is fresh; never start from BCE or smoke weights. Adam .001,
batch16, float32, max30 epochs, patience7, min_delta=.0001, original preprocessing,
paired training augmentation, patient split and size cutoffs remain identical.
Best checkpoint uses validation positive-slice patient-macro Dice at .5.

For each slice, p=sigmoid(logits), TP=sum(p*y), FP=sum(p*(1-y)), FN=sum((1-p)*y).

`loss = mean_slices[1 - (TP + 1e-6)/(TP + .3*FP + .7*FN + 1e-6)]`

Sum over channel and spatial dimensions, then average slices, including empty
masks. Smoothing is in pixel-sum units and appears in numerator and denominator.
This adopts the earlier proposed convention **before** definitive training.
No hybrid, weighting, focal term, batch-pooled index or postprocessing is added.
The [Tversky paper](https://arxiv.org/abs/1706.05721) motivates asymmetric FP/FN
weighting; smoothing and reduction here are explicit project choices.

For empty GT, the index is `s/(alpha*sum(p)+s)`. Increasing FP lowers the index;
the loss pushes logits downward, but with s=1e-6 and many FP pixels the gradient
can be extremely small. In float32 the loss can round to 1 while autograd still
returns small nonzero gradients. Saturated sigmoid can also suppress gradients.
This is a known limitation, not proof of a strong background-learning signal.
Inspect empty-mask FP outcomes honestly. Do not change this convention after
viewing validation outcomes; any later loss variant would be a separate experiment.
Evaluation metrics retain the unsmoothed, frozen definitions.

## Exact browser-Colab sequence

After a separately approved source commit/push, obtain the full SHA using
`git rev-parse HEAD`. Open:

`https://colab.research.google.com/github/hrb31/NeuroGuard/blob/COMMIT_SHA/notebooks/03_train_tversky.ipynb`

Replace COMMIT_SHA with the reviewed 40-character SHA. The repository is public;
no GitHub token is needed. Until source is pushed, uploading the notebook alone
cannot fetch its new modules. This implementation does not commit or push.

1. Select T4 and verify CUDA. Keep the installed CUDA PyTorch.
2. Enter that same full source SHA.
3. Clone the public repository; verify revision and clean checkout.
4. Install supporting dependencies (including `jedi`), run tests and strict
   `pip check`; verify PyTorch version/CUDA build did not change. Restart if
   imported supporting packages changed. Return genuine conflicts for review.
5. Mount Drive; enter `tversky-seed42-01` and NEW. Original BCE `identity.json`
   at `MyDrive/NeuroGuard/experiments/phase3-bce/bce-seed42-01/` is read-only.
   Check its config/core runtime and NumPy/Pillow/TIFF dependency versions;
   review recorded differences in other supporting packages.
6. Load the original manifest and its size cutoffs. The independent canonical
   identity is `7514044281736b609e9d1c09c342a3f2514d0a2ca4dea9c4ce0d466cd86fe493`.
   It is the canonical JSON identity, not a file-byte hash. Counts must match
   train77/2719/934 positive, validation16/621/209, test17/589/230. Never regenerate.
7. Review dataset/upstream terms and enter ACCEPT. Resolve pinned version2 inside
   Colab only; compare canonical-root copies and verify original file hashes.
   Test files receive integrity checking only; no test Dataset/predictions.
8. Verify development loaders, binary masks, unchanged model count and Tversky backward.
9. Inspect synthetic empty/tiny/large/background/perfect/mixed loss/gradient diagnostics.
10. Run 100 updates on four fixed positive training slices, inspect loss/Dice and
    overlays, then set `SMOKE_REVIEWED=True` in the separate gate cell. Review
    the empty-mask limitation and package differences. Stop on fitting failure.
11. Type TRAIN to run the definitive experiment. `fit()` resets seed42 and creates
    a fresh model/Adam; smoke state is discarded. OOM stops; do not change batch
    size for just this arm. No repeated seeds or extra budget.
12. Evaluate validation at .5 and the frozen threshold search; save detailed
    reports, size groups/counts/IoU, curves and FLAIR overlays at both thresholds.
    Compare measured results with supplied rounded BCE aggregates, in percentage
    points. Missing BCE metrics remain unavailable; no training-loss comparison.
13. Save provenance, config, environment, histories, reports, summaries and hashes;
    verify original manifest and BCE identity bytes are unchanged.
14. Return the review summary, pytest/smoke results, curves and representative
    overlays at both thresholds. Record negative findings and any failures.

The search remains .10,.15,...,.90: maximise positive-slice patient-macro Dice
with empty-mask FP <=10%; ties use lower FP then higher threshold. If none qualifies,
retain .5 and mark the constraint unmet. The 10% absolute and <=5 percentage-point
FP-increase research guardrails are reported, not described as clinical safety.
See [the unchanged evaluation rules](evaluation-protocol.md).

## Persistent artifacts and resume

Only `MyDrive/NeuroGuard/experiments/phase4-tversky/RUN_ID/` is a definitive run.
Timestamped preflight/smoke evidence goes under its sibling
`phase4-tversky/preflight/RUN_ID/TIMESTAMP/`. No Phase 2/3 artifact is written.

Keep `identity.json`, `config.json`, `environment.json`/`.txt`, `pytest.txt`,
`history.json`, two rotating state checkpoints with `latest.json`/`previous.json`,
`best.pt`/checksum, `validation_reports.json`, `validation_comparison.json`,
`bce_validation_reference.json`, `smoke_debug_only.json`, `loss_diagnostics.json`,
`package_differences_from_bce.json`, `training_curves.png`, `figures/`,
`review_summary.json` and `artifact_hashes.json`. Identity records source SHA,
original manifest/file hashes, dataset fingerprint, size cutoffs, model count,
runtime and BCE reference commit/identity checksum. Full state includes optimiser,
RNG, current/best models and patience state. Load only trusted private checkpoints.

After disconnect: SAME SHA, rerun 1–10, SAME run ID and RESUME, review smoke, 11–14.
Only an unfinished epoch is replayed; completed epochs count toward the same
30-epoch budget. Early-stopped runs do not gain more epochs. A matching identity
allows replay of epoch1 if interruption happened before the first complete save.
Config/commit/environment/checksum mismatch stops; never edit identity or manifest
to bypass it. CPU resume is tested locally; cloud GPU interruption recovery still
needs actual execution. Do not send patient records, tokens, MRI or weights back.

Phase 5 final evaluation and Gradio require later authorisation after results
review. The mandatory final code simplification, scientific-integrity review,
judge-facing notebook and submission audit remain deferred until core work is done.
OpenAI Codex assisted implementation; the student reviews, executes and explains it.

## Actual local verification — 2026-11-15

Python 3.14.2 / torch 2.15.0+cpu: 68 tests passed, two CUDA tests skipped in
40.50s; `pip check` found no broken requirements. Both training notebooks passed
their clean-output/compilation checks again after the final provenance-cell edit
(2 passed). These checks do not execute cloud notebook cells.

Separate unchanged full-U-Net CPU smoke: four generated 256x256 examples,
60 Tversky updates, loss .85148245096 to .55821371078, tiny pooled training
Dice .97516930023, exact checkpoint-logit reload. Plot visually inspected;
prediction closely matches the generated shape. Synthetic debugging only,
not MRI performance; artifacts remain ignored under `tmp/phase4-local-smoke/`.
Original Phase 2/3 notebooks, model, BCE config and data implementation have no
content diff; four original reference hashes match. Actual T4 training,
real-data smoke, cloud resume and Tversky validation remain unexecuted locally.
