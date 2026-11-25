# Empirical Experimental Results & Verification CSVs

This directory contains verified experimental log records produced by the rigorous 5-seed training and evaluation pipeline:

| CSV File | Corresponds to | Description |
|---|---|---|
| `rigorous_5seed_ablation_summary.csv` | **Table 7** in Paper | Systematic 5-seed component ablation breakdown (mean ± sample std dev, $ddof=1$, paired $t$-test $p$-values) |
| `cross_dataset_unified_results.csv` | **Table 5** in Paper | Zero-shot cross-dataset evaluation across 1,188 unseen clinical images (CVC-ClinicDB, CVC-ColonDB, ETIS-Larib) |
| `hardware_latency_benchmark.csv` | **Table 6** in Paper | Standardized latency, FPS, and peak VRAM profiling across batch sizes $B \in \{1, 4, 16\}$ on NVIDIA RTX 3090 GPU |
| `master_nextgen_experiments.csv` | **Table 3 & Master Archive** | Full raw logging of all 43 executed runs across all seeds, rungs, and configurations |

All numbers reported in the paper are derived directly from these CSV logs without manual post-processing.
