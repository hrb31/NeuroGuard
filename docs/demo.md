# Run NeuroGuard

This project is a prototype and is not medical advice. The intended user is a
radiologist or imaging researcher exploring model predictions on public research
images. Highlighted pixels are predictions, not confirmed abnormalities.

## Local installation and launch

From the repository root, using an existing Python environment with a compatible
CPU PyTorch build (Python 3.11 is a suggested portable starting point):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:7860**. CPU is the default. `--device cuda` is optional
when a working CUDA PyTorch installation already exists. Do not reinstall torch
inside the completed experiment environment. Local verification used Python
3.14.2, torch 2.15.0+cpu and Gradio 6.32.0; dependency ranges are not a universal lock.

Authentic private weights are expected at:

```text
local-assets/experiments/phase3-bce/bce-seed42-01/best.pt
local-assets/experiments/phase4-tversky/tversky-seed42-01/best.pt
```

Alternative locations can be passed with `--bce-checkpoint` and
`--tversky-checkpoint`. The app always checks the original SHA256 values in
[inference.py](../src/inference.py). There is no download, hash override, training,
threshold search or model substitute. Restart after replacing a checkpoint;
verified weights are cached in memory for the process lifetime.

## Input and output

Upload an **unmodified 256 × 256 × 3 uint8 LGG TIFF**, at most 4 MiB, one TIFF
page. Use the image, not its binary `_mask.tif`. Stored channels are pre-contrast,
FLAIR, post-contrast, with the dataset's documented substitutions. These are MRI
channels, not RGB colours. JPEG/PNG, DICOM, other sequences, volumes and arbitrary
photographs are unsupported. The app checks structure, not medical provenance;
the person uploading must confirm compatibility. Use only public, de-identified examples.

Select BCE 0.5, BCE 0.8 (validation-selected), or Tversky 0.5 (failed FP constraint,
fallback). Comparison modes run BCE/Tversky at the common thresholds or the
selected/fallback points. The original preprocessing is called directly: all
three stored channels, per-image/channel nonzero z-score, no augmentation.
Sigmoid probabilities are thresholded with `>=`; no cleanup is applied.

Outputs show actual uploaded FLAIR channel 1, a green prediction overlay and a
white-on-black binary mask. Output images use lossless PNG, preserving binary
mask and displayed pixel values. Cohort Dice is never an individual confidence score.
An error or changed input clears the preceding result.

## Application smoke test (not an experiment)

1. Verify the original weights are present; start `python app.py` on CPU.
2. Upload one reviewed original LGG TIFF. Confirm its provenance. Run BCE 0.5;
   inspect the actual FLAIR image, mask alignment and overlay.
3. Select each comparison mode; run again. Check both captions and the Tversky
   fallback warning. The binary-mask tab should correspond to the green overlay.
4. Try an unsupported PNG, a mask TIFF and a malformed file: the app must not
   interpret these as compatible scans. Run without confirmation; no output.
5. For an automated HTTP upload-to-output check on that same reviewed file:

```powershell
.\.venv\Scripts\python.exe tools/smoke_demo.py --scan "local-assets/demo-inputs/positive.tif"
.\.venv\Scripts\python.exe tools/smoke_demo.py --scan "local-assets/demo-inputs/empty.tif"
```

The script compares the returned images/masks with direct inference using the
same original weights and fixed points, and writes a **private application-only**
report. It calculates no accuracy metrics and reads no ground-truth masks or
patient manifest. This is not a repetition of the held-out evaluation.

Local verification completed on both genuine source TIFFs with the original
checksum-verified models: HTTP FLAIR/overlay/mask outputs matched direct CPU
inference exactly at all three frozen points. A real browser comparison was also
inspected. These two examples were chosen by fixed lexical source order and mask
presence before inference, not by performance. They are application smoke inputs,
not additional evaluation evidence. Original MRI and smoke screenshots remain private.

## Privacy and access

The server binds to loopback only, disables Gradio analytics and run history,
and creates no public tunnel. No model, dataset or remote API download occurs in
the application. Uploads and rendered images use ignored `tmp/gradio/`; Gradio
checks every minute for files older than five minutes. Close the app after use;
clear this cache before recording or handing over the machine. Do not expose this
local prototype on an untrusted network. Private directories are blocked from
Gradio file serving; that is additional protection, not a hosting security audit.

Public code and saved aggregate readouts can be understood without personal Drive
access. **Independent inference still requires authorised weights and a compatible
input; neither is bundled or publicly hosted.** The reliable fallback is a local
screen recording after a real smoke test. A small authorised model release could
remove the weights barrier later; I have not released weights or MRI examples, deployed the app, or enabled a
public tunnel. A recorded demonstration with specifically approved images is
the fastest practical viewing option; it does not require distributing weights.
