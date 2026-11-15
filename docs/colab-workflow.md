# VS Code -> GitHub -> browser Colab

The local repository is authoritative. Edit only
`notebooks/01_data_exploration.ipynb` and `src/` locally. Do not create another
exploration notebook or fix the VS Code Colab extension. The user's browser Colab
T4/CUDA check is reported successful; this phase has not remotely rerun it.

## Code delivery (requires approval before push)

The existing remote is `https://github.com/hrb31/NeuroGuard.git`. This work does
not push, publish or deploy. After reviewing and authorising delivery:

1. Review `git status`/diff. Stage only public code/config/docs and tests; keep
   `AGENTS.md`, `local-context/`, data, credentials and outputs excluded.
2. Commit the reviewed Phase 2 changes and push the approved commit.
3. Obtain the full commit SHA with `git rev-parse HEAD`.
4. Open `https://colab.research.google.com/github/hrb31/NeuroGuard/blob/COMMIT_SHA/notebooks/01_data_exploration.ipynb`,
   replacing COMMIT_SHA. For a private repository, use Colab's Open notebook ->
   GitHub, authorise browser access and select the repository/revision. Browser
   permission to view the notebook is separate from runtime clone authentication.
5. In notebook section 1 enter that same full SHA. Section 2 clones the repository,
   checks out that exact commit and makes its `src` modules importable.

Until a push is approved, a local-only preview is possible: upload the notebook
to Colab to read it. Its bootstrap intentionally cannot fetch unpushed source.
Do not run an old remote notebook and mistake it for these changes.

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
10. Save a private evidence ZIP and SHA256. Return the printed pytest output,
    summary/counts, review notes and figures to verify Phase 2. No training follows.

## Persistence and resets

Only raw dataset files/cache and code checkout are ephemeral. Reports, manifests,
figures, source metadata, preprocessing settings and environment go to Drive.
After reset, repeat sections 1-10 with the same commit and Drive folder. Section 9
re-audits files and verifies the existing immutable manifest before reusing it.
Code revisions get separate run directories; the dataset split remains shared and
frozen. Review warnings/errors instead of deleting a manifest to bypass them.
For this directory-resolution fix, use the updated notebook in a clean runtime
after the new code is approved and pushed. Rerun sections 1-8, return the root
comparison summary, audit counts and figures, then review before freezing. Old
wrapper-based audit reports remain historical evidence, not the canonical audit.

Colab source edits do not update the local checkout. If an experimental fix is
needed, apply it locally, review/commit/push with approval, and reopen at the new
commit in a clean runtime. Copying a notebook to Drive does not synchronise it to
GitHub. Before any public notebook submission, clear patient identifiers and
unreviewed image outputs; publish only a licence/privacy-reviewed presentation.

## Verified metadata versus pending evidence

On 2026-11-15 a live request to Kaggle's public
[dataset metadata endpoint](https://www.kaggle.com/api/v1/datasets/view/mateuszbuda/lgg-mri-segmentation)
returned version **2**, **CC BY-NC-SA 4.0**, last update 2019-05-02. Description:
TCIA images corresponding to TCGA lower-grade glioma patients and manual FLAIR
abnormality masks. Source citations include Buda, Saha & Mazurowski (2019), DOI
10.1016/j.compbiomed.2019.05.002, and Mazurowski et al. (2017). This verifies release
metadata only; file counts, channel meanings and mask integrity are pending Colab
execution. Attribution/noncommercial/share-alike obligations must be considered
before redistributing any data-derived artifacts, separately from a code licence.

The notebook uses the supported
[KaggleHub versioned download and secret authentication APIs](https://github.com/Kaggle/kagglehub).
Its actual cloud download and browser integration have not been executed here.
