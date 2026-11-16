# Sources, assistance and publication boundaries

## Dataset and concepts

- [Mateusz Buda, Brain MRI segmentation (LGG), version 2](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).
  Public Kaggle metadata rechecked 2026-11-15: **CC BY-NC-SA 4.0**. Source MRI
  provenance is TCIA/TCGA-LGG; FLAIR abnormality annotations are not independent
  diagnoses. Buda, Saha & Mazurowski (2019),
  [doi:10.1016/j.compbiomed.2019.05.002](https://doi.org/10.1016/j.compbiomed.2019.05.002).
- [Dataset author's project and channel description](https://mateuszbuda.github.io/2017/12/01/brainseg.html)
  and [upstream implementation](https://github.com/mateuszbuda/brain-segmentation-pytorch).
  These provide context, not NeuroGuard's measured results. NeuroGuard uses its
  own small local architecture and trained checkpoints, not the upstream weights.
- Ronneberger et al. (2015), [U-Net](https://arxiv.org/abs/1505.04597).
- Salehi et al. (2017), [Tversky loss](https://arxiv.org/abs/1706.05721).
- PyTorch [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html),
  [checkpoint loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
  and [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Tools and AI assistance

[PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/),
[Pillow](https://python-pillow.org/), [tifffile](https://github.com/cgohlke/tifffile),
[imagecodecs](https://github.com/cgohlke/imagecodecs),
[Matplotlib](https://matplotlib.org/), [Gradio](https://www.gradio.app/),
[Jupyter](https://jupyter.org/), [pytest](https://docs.pytest.org/),
[KaggleHub](https://github.com/Kaggle/kagglehub), [Requests](https://requests.readthedocs.io/),
[Google Colab](https://colab.research.google.com/), VS Code and Git/GitHub support
implementation, data access, plotting, tests and the demonstration. I used
KaggleHub for the Colab dataset acquisition and the separate local
application smoke inputs. The app does not fetch data or call a cloud inference service.

I developed NeuroGuard as an independent research project, directing the
experimental design, model comparison and analysis of the results. I used
[OpenAI ChatGPT](https://chatgpt.com/) and [Codex](https://openai.com/codex/)
to accelerate implementation, debugging, testing and documentation, allowing me
to focus on research decisions and interpretation. I ran and reviewed the genuine
Colab experiments. Local synthetic tests and application smoke tests are
engineering checks, not additional MRI performance evidence.

The supplied Veritas 2026 Project Briefs and ScamCheck example guided submission
structure. They are preserved privately, not redistributed here. The newer Briefs
override the older Guide: working code and an approximately one-page write-up are
required; a video of at most two minutes is optional but recommended.

## What may be public

Prepared source, methodology, aggregate tables and aggregate-only graphics are
intended for the reviewed public submission. No raw dataset, patient identifiers,
per-patient metrics, original private reports, checkpoints or MRI panels are
included in the proposed public additions. The public summary is a deliberately
curated aggregate extract with original report hashes, not the private run bundle.

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) permits sharing
and adaptation subject to attribution, noncommercial use, ShareAlike for adapted
material, notices of changes and its other terms. It does not waive privacy or
other rights. MRI examples/crops remain source-derived material: review each
specific export, give the dataset/paper/licence links, identify presentation
changes, apply the required terms, and obtain my explicit approval before sharing.
The existence of an accessible dataset is not unrestricted redistribution consent.

Whether a particular trained weight file is legally an adaptation is not resolved
by the dataset label alone. No blanket legal conclusion is made here. Checkpoint
licensing and redistribution terms need explicit review and my approval;
private checkpoints remain private. Deployment also requires this asset review.

I have not yet selected a repository-wide code licence. The upstream MIT licence
does not automatically license this repository, the dataset or its weights.
Third-party packages retain
their own licences. No external tutorial implementation was newly copied for the application;
the original source history remains available for provenance review.
