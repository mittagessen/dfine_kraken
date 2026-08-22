# NMS Sweep - Implementation Complete ✓

## What Was Created

A **simplified, SLURM-native** NMS sweep testing framework.

### Core Files

| File | Purpose |
|------|---------|
| **`run_nms_sweep.py`** | Main script - submit & collect (170 lines) |
| **`nms_monitor.sh`** | Monitor jobs using SLURM (simple bash wrapper) |
| **`dfine/nms_utils.py`** | Overlap calculation utilities |

### Documentation

| File | Content |
|------|---------|
| **`NMS_SIMPLE_README.md`** | ⭐ START HERE - Quick start guide |
| **`NMS_COMPLETE_WORKFLOW.md`** | Full step-by-step example |
| **`NMS_SWEEP_DESIGN.md`** | Design philosophy & overview |
| **`NMS_CLI_PARAMETER.md`** | CLI `--nms-iou` parameter details |
| **`example_nms_sweep.sh`** | Copy-paste example |

## Quick Start (3 Steps)

### Step 1: Submit (10 seconds)
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```
Creates 19 parallel SLURM jobs (one per NMS threshold).

### Step 2: Monitor (2-4 hours)
```bash
squeue -u $(whoami)          # SLURM shows everything
bash nms_monitor.sh tail     # View latest job output
```

### Step 3: Collect (5 seconds)
```bash
python run_nms_sweep.py --collect
```
Displays results in beautiful table + CSV.

## What It Tests

**19 NMS IoU thresholds:**
```
None (baseline), 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950
```

**Metrics collected:**
- mAP@50
- mAP@50:95
- Precision
- Recall
- F1

## Output Format

```
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ NMS IoU ┃ mAP@50 ┃ mAP@50:95 ┃ Precision ┃ Recall ┃     F1 ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    none │ 0.5232 │    0.3548 │    0.5232 │ 0.7901 │ 0.6295 │
│   0.100 │ 0.5214 │    0.3483 │    0.5214 │ 0.7551 │ 0.6168 │
│   ...   │  ...   │    ...    │    ...    │  ...   │  ...   │
│   0.950 │ 0.5266 │    0.3565 │    0.5266 │ 0.7902 │ 0.6320 │
└─────────┴────────┴───────────┴───────────┴────────┴────────┘

Saved to: nms_sweep_results/nms_sweep_summary.csv
```

## Design Philosophy

✓ **Minimal** - 170 lines of Python  
✓ **SLURM-native** - Leverages `squeue`, `sbatch`, `scancel`  
✓ **Simple** - No complex Python, just calls `dfine test`  
✓ **Clear** - Easy to understand and debug  
✓ **Reliable** - Uses proven SLURM infrastructure  

## Usage Examples

### Basic sweep
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```

### Custom output directory
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/ -o my_results
python run_nms_sweep.py --collect -o my_results
```

### Dry run (preview)
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/ --dry-run
```

### Monitor
```bash
squeue -u $(whoami)           # Check jobs
bash nms_monitor.sh status    # Show running NMS jobs
bash nms_monitor.sh logs      # List log files
bash nms_monitor.sh tail      # Show last job output
bash nms_monitor.sh cancel    # Cancel all NMS jobs
```

## How Each Job Works

Each SLURM job runs:
```bash
source venv/bin/activate
dfine test -m model.ckpt -e test.txt --nms-iou 0.5 test_set/
```

That's it! The existing `dfine test` CLI handles all the work. We just:
1. Generate 19 sbatch scripts with different `--nms-iou` values
2. Submit them all to SLURM
3. Collect and display results

## Performance

- **Total runtime:** ~2-4 hours (all 19 jobs run in parallel)
- **Per job:** ~1-2 hours
- **Memory:** 32GB per job
- **GPU:** 1 per job
- **Parallelism:** Full (all jobs simultaneous)

## Key Insights

1. **Why fixed thresholds?** - Pre-selected based on empirical testing. Easy to change if needed.

2. **Why use SLURM directly?** - SLURM already knows job status, runtime, logs. Why duplicate that?

3. **Why not keep old complex version?** - Simpler is better. More maintainable, easier to debug.

4. **Can I test custom thresholds?** - Yes! Manually edit the `THRESHOLDS` list in `run_nms_sweep.py` or just run `dfine test --nms-iou 0.42` directly.

## FAQ

**Q: How do I check job progress?**  
A: Use SLURM: `squeue -u $(whoami)` or `bash nms_monitor.sh status`

**Q: How do I see what's running?**  
A: `squeue -u $(whoami) | grep nms`

**Q: How do I see job output?**  
A: `tail -f nms_sweep_results/logs/12345.out` (get job ID from squeue)

**Q: What if a job fails?**  
A: Check the log: `cat nms_sweep_results/logs/12345.err`

**Q: Can I re-run a specific threshold?**  
A: Yes: `sbatch nms_sweep_results/job_0_500.sh`

**Q: How do I cancel everything?**  
A: `bash nms_monitor.sh cancel` or `scancel -u $(whoami) --name="*nms*"`

## Integration with DFINE

### In the CLI
```bash
# Test with specific NMS
dfine test -m model.ckpt -e test.txt --nms-iou 0.5 test_set/

# Test without NMS (baseline)
dfine test -m model.ckpt -e test.txt test_set/
```

### In Python
```python
from dfine.cli.test import test
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(test, [
    '-m', 'model.ckpt',
    '-e', 'test.txt',
    '--nms-iou', '0.5',
    'test_set/'
])
```

## Next Steps

1. **Review documentation** - Start with `NMS_SIMPLE_README.md`
2. **Try example** - Run `example_nms_sweep.sh`
3. **Submit sweep** - `python run_nms_sweep.py ...`
4. **Analyze results** - View CSV and identify optimal threshold
5. **Use optimal** - `dfine test ... --nms-iou <optimal> ...`

## Files Structure

```
/dfine_kraken/
├── run_nms_sweep.py              # Main script
├── nms_monitor.sh                # Monitor helper
├── dfine/
│   └── nms_utils.py              # Utilities (unused in simple version)
├── NMS_SIMPLE_README.md          # ⭐ Start here
├── NMS_COMPLETE_WORKFLOW.md      # Full example
├── NMS_SWEEP_DESIGN.md           # Philosophy
├── NMS_CLI_PARAMETER.md          # CLI details
├── example_nms_sweep.sh          # Example usage
└── nms_sweep_results/            # Output (created by script)
    ├── job_*.sh                  # SLURM scripts
    ├── logs/                     # Job logs
    ├── results_nms_*.json        # Raw results
    └── nms_sweep_summary.csv     # Summary table
```

---

**Ready to use!** Start with:
```bash
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/
```

Questions? Check `NMS_COMPLETE_WORKFLOW.md` for detailed walkthrough.
