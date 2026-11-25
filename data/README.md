# Datasets for MS-VMUNet

This repository supports four publicly available clinical colonoscopy benchmark datasets:

## 1. Primary Benchmark: Kvasir-SEG
- **Source**: Vestre Viken Health Trust, Norway (Jha et al., MMM 2020)
- **Samples**: 1,000 polyp images with ground-truth binary masks
- **Partitioning Protocol**: Strict 80:10:10 split (800 training, 100 validation, 100 test images), fixed deterministically via Seed 42.

## 2. Multi-Center Zero-Shot Generalization Corpora
- **CVC-ClinicDB**: 612 standard mucosal colonoscopy frames (Hospital Clínic de Barcelona, Bernal et al., 2015)
- **CVC-ColonDB**: 380 challenging frames with vascular folds and varied morphologies (Tajbakhsh et al., 2015)
- **ETIS-Larib**: 196 small, early-stage micro-polyps (Lariboisière Hospital, Silva et al., 2014)

---

### One-Click Download:
Run the automated downloader script:
```bash
bash data/download_datasets.sh
```

Directory structure after download:
```text
data/
├── Kvasir-SEG/
│   ├── images/
│   └── masks/
└── PolyDataset/
    ├── CVC-ClinicDB/
    │   ├── images/
    │   └── masks/
    ├── CVC-ColonDB/
    │   ├── images/
    │   └── masks/
    └── ETIS/
        ├── images/
        └── masks/
```
