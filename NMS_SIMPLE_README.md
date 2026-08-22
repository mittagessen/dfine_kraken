# NMS Sweep - Simple Parallel Testing

Test multiple NMS IoU thresholds in parallel using SLURM.

## Quick Start

### 1. Submit all NMS sweep jobs (19 parallel jobs)
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```

### 2. Monitor with SLURM (real-time)
```bash
# Check running jobs
squeue -u $(whoami)

# View recent logs
bash nms_monitor.sh logs
bash nms_monitor.sh tail

# Cancel all
bash nms_monitor.sh cancel
```

### 3. Collect results (when done)
```bash
python run_nms_sweep.py --collect
```

## Output

Beautiful ASCII table showing metrics for each NMS threshold + CSV file.

## What It Tests

NMS IoU thresholds (19 total):
- **none** (baseline, no NMS)
- 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500
- 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950

## Files

- `run_nms_sweep.py` - Main script (submit & collect)
- `nms_monitor.sh` - Monitor jobs (simple bash wrapper)
- `nms_sweep_results/` - Output directory
  - `job_*.sh` - SLURM scripts
  - `logs/` - Job output/errors
  - `results_nms_*.json` - Raw results
  - `nms_sweep_summary.csv` - Summary table

## Tips

- Each job: 2 hours, 32GB, 1 GPU
- All jobs run in parallel
- Use `squeue` directly to monitor (SLURM knows everything)
- Logs are in `nms_sweep_results/logs/`
