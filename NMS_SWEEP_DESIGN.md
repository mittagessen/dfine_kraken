# NMS Sweep Implementation - Simplified for SLURM

## Overview

Created a **simple, minimal** NMS sweep system that leverages SLURM for job management instead of reinventing the wheel.

## Files Created

### Main Scripts

1. **`run_nms_sweep.py`** (main script)
   - Submit: `python run_nms_sweep.py -m model.ckpt -e test.txt test_set/`
   - Collect: `python run_nms_sweep.py --collect`
   - Just 170 lines - simple and clear

2. **`nms_monitor.sh`** (SLURM wrapper)
   - Status: `bash nms_monitor.sh status` (shows squeue)
   - Logs: `bash nms_monitor.sh logs` (list log files)
   - Tail: `bash nms_monitor.sh tail` (show last job output)
   - Cancel: `bash nms_monitor.sh cancel` (scancel all NMS jobs)

### Documentation

3. **`NMS_SIMPLE_README.md`** - Quick start guide
4. **`example_nms_sweep.sh`** - Example usage script

## How It Works

### Step 1: Submit (one command)
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```

Creates 19 sbatch scripts (one per NMS threshold) and submits them all to SLURM. Done in seconds.

### Step 2: Monitor (use SLURM directly)
```bash
squeue -u $(whoami)           # Check which jobs are running
bash nms_monitor.sh tail      # View last job's output
tail -f nms_sweep_results/logs/12345.out  # Follow specific job
```

SLURM already knows:
- Job IDs
- Runtime
- Memory usage
- Status (running/completed/failed)
- Output files

### Step 3: Collect (one command)
```bash
python run_nms_sweep.py --collect
```

Reads all `results_nms_*.json` files and displays a nice table + saves CSV.

## Why This Design

**Principle**: Don't duplicate what SLURM already does perfectly.

- ✗ Custom job tracking - SLURM already has `squeue`
- ✗ Custom progress bar - SLURM already shows runtime
- ✗ Custom log rotation - SLURM already handles logs
- ✗ Complex Python - Just call `dfine test` directly

**Result**: 
- Simple to understand
- Easy to debug
- Minimal dependencies
- Leverages familiar SLURM tools

## Thresholds

Fixed 19 thresholds (empirically chosen):
```
None, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950
```

## Usage Examples

### Example 1: Basic sweep
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```

### Example 2: Custom output directory
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/ -o my_results
python run_nms_sweep.py --collect -o my_results
```

### Example 3: Dry run (preview without submitting)
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/ --dry-run
```

### Example 4: Monitor jobs
```bash
# Terminal 1: Submit
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/

# Terminal 2: Monitor
watch squeue -u $(whoami)
bash nms_monitor.sh tail
```

## Output Files

```
nms_sweep_results/
├── job_none.sh              # SLURM script for baseline
├── job_0_100.sh             # SLURM script for threshold 0.100
├── ...                      # More scripts
├── logs/
│   ├── 12345.out           # Job 12345 output
│   ├── 12346.out           # Job 12346 output
│   └── ...
├── results_nms_none.json    # Raw results from each job
├── results_nms_0_100.json
├── ...
└── nms_sweep_summary.csv    # Final summary table
```

## Key Commands to Remember

```bash
# Submit all 19 jobs
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/

# Check status (SLURM native)
squeue -u $(whoami)

# Monitor logs
bash nms_monitor.sh tail

# Collect results
python run_nms_sweep.py --collect

# Cancel all jobs
bash nms_monitor.sh cancel
```

## Performance

- **Total runtime**: ~2-4 hours (all 19 jobs run in parallel)
- **Per job**: ~1-2 hours
- **Memory**: 32GB per job
- **GPUs**: 1 per job

## Dependencies

- Python 3.6+
- SLURM (sbatch, squeue, scancel)
- dfine (already installed in venv)

## Troubleshooting

**Jobs not running?**
```bash
squeue -u $(whoami)  # Check if submitted
ls nms_sweep_results/logs/ | head  # Check for errors
```

**Want to see which threshold is running?**
```bash
squeue -u $(whoami) | grep nms
```

**Check specific job output?**
```bash
cat nms_sweep_results/logs/12345.out
```

**Cancel everything?**
```bash
bash nms_monitor.sh cancel
# Or: scancel -u $(whoami) --name="*nms*"
```

## Integration with Your Workflow

The sweep creates one JSON result file per job. You can:

1. Parse the JSON files programmatically
2. Plot metrics vs NMS threshold
3. Find optimal threshold for your use case
4. Use that threshold for production inference

All raw data is in `results_nms_*.json` for custom analysis.
