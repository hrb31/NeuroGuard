# Reusable Python modules

Phase 2 modules: acquisition.py (metadata, Colab-only pinned download and canonical
root resolution with full TIFF-copy hash comparison), data.py
(TIFF audit and immutable patient split), preprocessing.py (paired transforms),
visualization.py (training-only Phase 2 previews). Phase 2 data verification was
completed in user Colab execution. Phase 3 adds model.py (small logit U-Net),
dataset.py (read-only manifest and seeded train/validation loaders), metrics.py
(patient-macro/empty/size/threshold protocol), training.py (BCE epochs and verified
resume), smoke.py (train-only debugging), reporting.py (private validation figures).
Phase 4 adds losses.py (explicit BCE/Tversky factory and soft per-slice Tversky),
shared fit/smoke criterion selection, loss-aware curves and aggregate validation
comparison in reporting.py. The model, original BCE config and checkpoint schema
are unchanged. No final test evaluation or application inference is implemented.
