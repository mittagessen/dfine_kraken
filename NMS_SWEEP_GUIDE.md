# NMS Sweep Testing Guide

This guide explains how to run NMS sweep tests in parallel using sbatch to test different NMS IoU thresholds and measure overlapping zones.

## Overview

The NMS sweep script (`run_nms_sweep.py`) will:
1. Generate NMS IoU thresholds from 0.0 to 0.92 in steps of 0.05 (19 thresholds by default)
2. Create sbatch scripts for each threshold that run `dfine test --nms-iou <value>`
3. Submit all jobs to the cluster in parallel
4. Collect results and generate a summary table with metrics for each threshold

## Prerequisites

- Access to an SLURM cluster with sbatch
- A trained D-FINE model
- Test data manifest file
- Test set directory

## Basic Usage

### 1. Submit all NMS sweep jobs

```bash
python run_nms_sweep.py \
    -m /path/to/model.ckpt \
    -e /path/to/test_manifest.txt \
    /path/to/test_set
```

This will:
- Create `nms_sweep_results` directory
- Generate sbatch scripts for each NMS threshold (0.0, 0.05, 0.10, ..., 0.92)
- Each script runs: `dfine test -m model.ckpt -e test.txt --nms-iou <threshold> test_set`
- Submit all jobs to the cluster
- Print job IDs for tracking

### 2. Monitor jobs

Check job status while running:
```bash
squeue -u $(whoami)
```

Check logs:
```bash
tail -f nms_sweep_results/logs/*.out
```

### 3. Collect results

Once all jobs complete, collect and display results:
```bash
python run_nms_sweep.py --collect
```

Or specify a custom output directory:
```bash
python run_nms_sweep.py --collect -o my_results
```

This will:
- Read all result JSON files
- Create a summary table with metrics for each NMS threshold
- Save a CSV file with all results
- Print best configurations by different metrics

## Advanced Options

### Custom threshold range

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test_manifest.txt \
    test_set \
    --start 0.0 \
    --end 0.95 \
    --step 0.05
```

### Dry run (preview without submitting)

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test_manifest.txt \
    test_set \
    --dry-run
```

This will show which jobs would be submitted without actually submitting them.

### Custom output directory

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test_manifest.txt \
    test_set \
    -o my_nms_sweep_results
```

## Output Files

After jobs complete, you'll find:

```
nms_sweep_results/
├── logs/
│   ├── <job_id>.out
│   └── <job_id>.err
├── sbatch_nms_*.sh          # Individual sbatch scripts
├── results_nms_*.json       # Results for each threshold
├── predictions_nms_*.pkl    # Predictions (if saving enabled)
└── nms_sweep_summary.csv    # Summary table (CSV)
```

## Summary Output

The summary table will show:

| NMS IoU | Overlaps | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---------|----------|--------|-----------|-----------|--------|-----|
| 0.0     | 145      | 0.6234 | 0.5421    | 0.8123    | 0.7345 | 0.77|
| 0.05    | 132      | 0.6245 | 0.5435    | 0.8145    | 0.7356 | 0.77|
| ...     | ...      | ...    | ...       | ...       | ...    | ... |

Best configurations by metric will also be displayed.

## Result Interpretation

- **NMS IoU**: The IoU threshold used for Non-Maximum Suppression
- **Overlaps**: Number of overlapping detection zones (lower is better for filtering)
- **mAP@50**: Mean Average Precision at IoU=0.50
- **mAP@50:95**: Mean Average Precision averaged over IoU thresholds 0.50-0.95
- **Precision**: Detection precision
- **Recall**: Detection recall
- **F1**: Harmonic mean of precision and recall

## Notes

- Each job requires ~1-2 hours and 32GB GPU memory
- Jobs run in parallel on the cluster
- To cancel all jobs: `scancel -u $(whoami)`
- Logs are saved in `nms_sweep_results/logs/`
- Results are saved as JSON files for easy parsing

## Example Workflow

```bash
# 1. Submit NMS sweep jobs
python run_nms_sweep.py -m model.ckpt -e test.txt test_dir

# 2. Check progress (while jobs run)
squeue -u $(whoami)

# 3. Monitor logs
tail -f nms_sweep_results/logs/*.out

# 4. Once all complete, collect results
python run_nms_sweep.py --collect

# 5. Analyze CSV file
cat nms_sweep_results/nms_sweep_summary.csv
```

## Troubleshooting

**Jobs not submitting:**
- Check that sbatch is available: `which sbatch`
- Check model and test data paths exist
- Check that venv activation path is correct

**Jobs failing:**
- Check logs: `cat nms_sweep_results/logs/<job_id>.err`
- Verify model path and test data
- Check GPU availability

**No results collected:**
- Check that jobs have finished: `squeue -u $(whoami)`
- Verify result files exist: `ls nms_sweep_results/results_nms_*.json`
- Check for errors in job logs

## Contact

For issues or questions about the NMS sweep script, please refer to the test module documentation or check the job logs.
