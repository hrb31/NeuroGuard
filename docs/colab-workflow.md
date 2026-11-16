# Original data-audit workflow

I completed the data audit and froze the patient split before training. This is
the historical procedure for [01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb),
retained for reproducibility. The completed results and app have a separate
[reproduction guide](reproduction.md); the original experiments are not rerun
as part of the final notebook or application demonstration.

The original notebook opens a reviewed full Git SHA in browser Colab, clones that
same revision and imports its source modules. A GPU-enabled Colab environment
was used for later experiments. Public GitHub source requires no access token.
The private-repository authentication branch remains documented in the original
notebook, but is unnecessary for this public repository.

## Run the numbered notebook sections

1. Confirm environment; enter the exact reviewed commit.
2. Clone and verify source. For private repos set PRIVATE_REPOSITORY=True and
   provide read-only `GITHUB_TOKEN` through Colab Secrets or hidden prompt. Never
   embed credentials in a notebook, repository URL or output. Git uses a temporary
   askpass helper; the token is removed from the subprocess environment afterward.
3. Install `requirements-audit.txt` only and run synthetic tests. Preserve the
   existing working GPU PyTorch installation. If packages were already imported
   before installation, restart the runtime and rerun with the same commit.
4. Mount Drive and grant access. Outputs go to private
   `MyDrive/NeuroGuard/dataset-v2-seed42/`. Record the environment and code revision.
5. Refresh Kaggle metadata, read licence/upstream terms, and type ACCEPT only after
   reviewing them. If metadata is inaccessible, stop and return the error; never
   substitute guessed licence/version values.
6. Download version 2 via KaggleHub, **inside Colab only**. If required, set a
   `KAGGLE_API_TOKEN` Colab secret from https://www.kaggle.com/settings/api or use
   `kagglehub.login()` with its hidden prompt. Do not send tokens back for debugging.
7. Resolve the canonical dataset directory, then audit and print proposed split
   counts. Kaggle's extracted wrapper may include both `kaggle_3m/` and
   `lgg-mri-segmentation/kaggle_3m/`. The resolver compares every relative TIFF
   filename and SHA256 hash in each copy (images AND masks). Only identical copies
   allow selection of one root, preferring the shallowest path then lexical order.
   Missing, extra or changed TIFFs cause an integrity error, not silent selection.
   Unassigned TIFFs and overlapping roots also require review. No raw files are
   deleted/changed. `root_resolution.json` saves candidate paths and per-file hashes.
   All subsequent audit, previews/preprocessing and freeze operations use that same
   `DATA_ROOT`. Duplicate detection inside it remains unchanged.
8. Review training-only channel images, masks, overlays and processed examples.
   Consult release documentation to establish channel meanings. Channel equality
   and appearance alone cannot prove modalities or FLAIR channel position.
   Pause here for review before running section 9. Expected unique counts are
   110 patients, 3,929 images/masks, 1,373 positive and 2,556 empty slices; these
   remain expectations until the corrected real-data audit calculates them. Do
   not modify counts or pick a root because its totals match these expectations.
9. Add honest review notes and freeze the split. Any missing/corrupted pair blocks
   freezing. Cross-patient nonblank image duplicates require resolution and a
   fresh audit. All-zero background duplicates may be explicitly reviewed. Never
   change a seed because a model performs badly. Existing manifests cannot be
   overwritten with different assignments.
10. Save a private evidence ZIP and SHA256. Retain the printed pytest output, summary/counts, review notes and figures
    with the private audit evidence.

## Persistence and resets

Only raw dataset files/cache and code checkout are ephemeral. Reports, manifests,
figures, source metadata, preprocessing settings and environment go to Drive.
After reset, repeat sections 1-10 with the same commit and Drive folder. Section 9
re-audits files and verifies the existing immutable manifest before reusing it.
Code revisions get separate run directories; the dataset split remains shared and
frozen. Review warnings/errors instead of deleting a manifest to bypass them.
## Phase 2 metadata and historical evidence gate

On 2026-11-15 a live request to Kaggle's public
[dataset metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/mateuszbuda/lgg-mri-segmentation)
returned version **2**, **CC BY-NC-SA 4.0**, last update 2019-05-02. Description:
TCIA images corresponding to TCGA lower-grade glioma patients and manual FLAIR
abnormality masks. Source citations include Buda, Saha & Mazurowski (2019), DOI
10.1016/j.compbiomed.2019.05.002, and Mazurowski et al. (2017). This verifies release
metadata only. Subsequent counts, channel meanings and integrity were verified
in my Colab audit as recorded in the final notebook. The original workflow
above describes the evidence gates at that time. Attribution/noncommercial/share-alike obligations must be considered
before redistributing any data-derived artifacts, separately from a code licence.

The notebook uses the supported
[KaggleHub versioned download and secret authentication APIs](https://github.com/Kaggle/kagglehub).
I completed the actual data acquisition and audit in Colab; local synthetic
checks alone do not establish dataset integrity or model performance.
