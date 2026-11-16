# Reproduction and frozen experimental evidence

The final results can be **read and replotted without personal Drive access**:
open [the final notebook](../notebooks/NeuroGuard_Final_Submission.ipynb).
Its saved outputs come from actually executed aggregate-readout cells. Optional
re-execution needs `requirements-notebook.txt` and reads only
[final-summary.json](../results/final-summary.json); it loads no models or MRI.

For inference use [demo.md](demo.md). Authentic authorised checkpoints and
compatible inputs are separate private prerequisites. Public source alone is
not a fully independent reproduction of private training/evaluation artifacts.
No public weights, sample MRI or live deployment is implied.

## Completed research record

The original four notebooks and configurations are preserved unchanged:

| Stage | Notebook | Method/provenance |
|---|---|---|
| Data audit and frozen split | [01](../notebooks/01_data_exploration.ipynb) | [Colab/data workflow](colab-workflow.md) |
| BCE | [02](../notebooks/02_train_bce.ipynb) | [Training](phase3-training.md), [validation findings](phase3-validation.md) |
| Matched Tversky | [03](../notebooks/03_train_tversky.ipynb) | [Training](phase4-training.md), [validation findings](phase4-validation.md) |
| Locked held-out evaluation | [04](../notebooks/04_final_evaluation.ipynb) | [Original preflight/gate](phase5-evaluation.md) |

I preserve these notebooks as the original experimental record. They are not
part of the final results-readout or application smoke test. Their historical
execution-gate wording is unchanged.
The earlier validation reports describe their state at the time; use the final
summary for completed held-out findings. My private run bundles preserve
environment, histories, manifest, source/config hashes, checkpoints and reports.
No split regeneration, retraining, validation search or test rerun is part of the
submission readout or demonstration. Original source and selection rules are in
[evaluation-protocol.md](evaluation-protocol.md) and [configs](../configs/).

Final evaluation source: `d26811abb564680075030d7970633b4e681a98d4`.
Manifest: `7514044281736b609e9d1c09c342a3f2514d0a2ca4dea9c4ce0d466cd86fe493`.

| Checkpoint | Original SHA256 |
|---|---|
| BCE | `ba0ce014980ecac8e8670357f2f2eff2484c719e1c2a7080ec5367dc85701eda` |
| Tversky | `1828ab28fe7077923447d6fe3f31a297173b14b93db6323ad68f9349bfe95708` |

Both original files exported for local development matched these hashes. Identical
configurations aside from experiment identity/loss, original architecture and
preprocessing are retained. A single split/seed is not training-seed robustness.

## Local engineering checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests generate synthetic data/weights; they do not measure MRI performance or
rerun the experiments. CUDA-specific tests skip explicitly on CPU. Frozen-phase
test code may exercise training/evaluation functions on tiny **synthetic** data.
The original four notebooks are never executed by these checks.
