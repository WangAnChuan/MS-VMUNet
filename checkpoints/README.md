# Pre-trained Model Checkpoints

This directory contains pre-trained weights for the proposed **MS-VMUNet-Lite (1.85M params)**:

| Checkpoint File | Seed | Test Dice (%) | Test IoU (%) | HD95 (px) | Description |
|---|:---:|:---:|:---:|:---:|---|
| `nextgen_full_best_seed200.pth` | 200 | **88.96%** | **82.20%** | **24.48** | Best-performing single model checkpoint (Table 3) |
| `nextgen_full_best_seed42.pth` | 42 | **87.01%** | **80.32%** | **24.83** | Benchmark reference model used for cross-dataset generalization (Table 5) |

### How to Evaluate
To evaluate a checkpoint on Kvasir-SEG test set:
```bash
python evaluate.py --checkpoint checkpoints/nextgen_full_best_seed200.pth
```

To evaluate a checkpoint on the 3 external clinical datasets (CVC-ClinicDB, CVC-ColonDB, ETIS-Larib):
```bash
python eval_cross_dataset_unified.py --checkpoint checkpoints/nextgen_full_best_seed42.pth
```
