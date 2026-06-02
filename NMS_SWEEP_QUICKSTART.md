# Quick Start: NMS Sweep Testing

## What it does

Tests your D-FINE model with 19 predefined NMS IoU thresholds **all in parallel** and creates a summary table comparing results.

**Default thresholds:** None (baseline), 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950

## Quick Start (3 steps)

### 1. Submit all NMS sweep jobs

```bash
python run_nms_sweep.py \
    -m /path/to/model.ckpt \
    -e /path/to/test_manifest.txt \
    /path/to/test_set
```

This runs 19 tests in parallel, each with a different NMS threshold.

### 2. Wait for completion (check status)

```bash
# Check which jobs are running
squeue -u $(whoami)

# Monitor progress
python nms_monitor.py progress

# View latest logs
python nms_monitor.py logs
```

### 3. Collect results

Once all jobs complete:

```bash
python run_nms_sweep.py --collect
```

This creates a nice summary table showing metrics for each NMS threshold.

## Example with real paths

```bash
# Submit jobs
python run_nms_sweep.py \
    -m models/dfine_model.ckpt \
    -e data/test_manifest.txt \
    data/test_images

# Wait ~2-4 hours for jobs to complete...

# Collect results
python run_nms_sweep.py --collect
```

## What you'll get

A summary table like:

```
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ NMS IoU ┃ mAP@50 ┃ mAP@50:95 ┃ Precision ┃ Recall ┃     F1 ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    none │ 0.5232 │    0.3548 │    0.5232 │ 0.7901 │ 0.6295 │
│   0.100 │ 0.5214 │    0.3483 │    0.5214 │ 0.7551 │ 0.6168 │
│   0.150 │ 0.5228 │    0.3490 │    0.5228 │ 0.7600 │ 0.6194 │
│   0.200 │ 0.5233 │    0.3493 │    0.5233 │ 0.7652 │ 0.6216 │
│   ...   │  ...   │    ...    │    ...    │  ...   │  ...   │
│   0.950 │ 0.5266 │    0.3565 │    0.5266 │ 0.7902 │ 0.6320 │
└─────────┴────────┴───────────┴───────────┴────────┴────────┘
```

## Manual test (single threshold)

If you just want to test one threshold instead of sweeping:

```bash
dfine test -m model.ckpt -e test.txt --nms-iou 0.5 test_set
```

## Options

### Custom threshold range

Test from 0.0 to 0.95 with 0.1 steps:

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test.txt \
    test_set \
    --start 0.0 --end 0.95 --step 0.1
```

### Custom output directory

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test.txt \
    test_set \
    -o my_results
```

Then collect with:
```bash
python run_nms_sweep.py --collect -o my_results
```

### Dry run (preview without submitting)

```bash
python run_nms_sweep.py \
    -m model.ckpt \
    -e test.txt \
    test_set \
    --dry-run
```

## Troubleshooting

**Jobs not submitting?**
```bash
# Check sbatch is available
which sbatch

# Check model and data exist
ls model.ckpt test.txt test_set/
```

**Need to cancel all jobs?**
```bash
scancel -u $(whoami)
```

**Want to see job logs?**
```bash
# Check all logs
ls nms_sweep_results/logs/

# View specific log
tail -f nms_sweep_results/logs/12345.out
```

**Jobs finished but --collect shows no results?**
```bash
# Check if result files exist
ls nms_sweep_results/results_nms_*.json

# Check job errors
ls nms_sweep_results/logs/*.err
```

## Files created

- `nms_sweep_results/` - Output directory
  - `sbatch_nms_*.sh` - Individual job scripts
  - `results_nms_*.json` - Results for each threshold
  - `nms_sweep_summary.csv` - Summary table
  - `logs/` - Job output/error logs

## More help

For detailed info, see:
- `NMS_SWEEP_GUIDE.md` - Complete user guide
- `NMS_SWEEP_IMPLEMENTATION.md` - Technical details

## TL;DR

```bash
# Run sweep
python run_nms_sweep.py -m model.ckpt -e test.txt test_set

# Wait for jobs...
python nms_monitor.py progress

# Get results
python run_nms_sweep.py --collect
```

Done! Check `nms_sweep_results/nms_sweep_summary.csv` for the results table.
