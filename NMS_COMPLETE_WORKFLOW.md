# NMS Sweep Workflow - Step by Step

## Complete Example

### Prerequisites
- Model file: `/models/dfine_model.ckpt`
- Test manifest: `/data/test_manifest.txt`
- Test images: `/data/test_images/`

### Workflow

#### 1. Submit All Jobs (10 seconds)

```bash
cd /home/jjanes/Documents/4_LADaS/dfine_kraken

python run_nms_sweep.py \
    -m /models/dfine_model.ckpt \
    -e /data/test_manifest.txt \
    /data/test_images/
```

**Output:**
```
Submitting 19 NMS sweep jobs...
Model: /models/dfine_model.ckpt
Test data: /data/test_manifest.txt
Test set: /data/test_images/

  [ 1] NMS=   none → Job 12345
  [ 2] NMS=  0.100 → Job 12346
  [ 3] NMS=  0.150 → Job 12347
  ...
  [19] NMS=  0.950 → Job 12363

✓ Submitted 19 jobs
Check status: squeue -u $(whoami)
Results: nms_sweep_results/nms_sweep_summary.csv
```

#### 2. Wait & Monitor (2-4 hours)

**Terminal 1 - Monitor jobs:**
```bash
# Watch all jobs
watch squeue -u $(whoami)

# Or just NMS jobs
squeue -u $(whoami) | grep nms

# Check how many completed
ls nms_sweep_results/logs/ | wc -l
```

**Terminal 2 - View latest output:**
```bash
bash nms_monitor.sh tail
```

**Terminal 3 - Follow specific job:**
```bash
tail -f nms_sweep_results/logs/12345.out
```

#### 3. Collect Results (5 seconds)

Once all jobs complete:

```bash
python run_nms_sweep.py --collect
```

**Output:**
```
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ NMS IoU ┃ mAP@50 ┃ mAP@50:95 ┃ Precision ┃ Recall ┃     F1 ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│    none │ 0.5232 │    0.3548 │    0.5232 │ 0.7901 │ 0.6295 │
│   0.100 │ 0.5214 │    0.3483 │    0.5214 │ 0.7551 │ 0.6168 │
│   0.150 │ 0.5228 │    0.3490 │    0.5228 │ 0.7600 │ 0.6194 │
│   0.200 │ 0.5233 │    0.3493 │    0.5233 │ 0.7652 │ 0.6216 │
│   0.250 │ 0.5241 │    0.3496 │    0.5241 │ 0.7687 │ 0.6232 │
│   ...   │  ...   │    ...    │    ...    │  ...   │  ...   │
│   0.950 │ 0.5266 │    0.3565 │    0.5266 │ 0.7902 │ 0.6320 │
└─────────┴────────┴───────────┴───────────┴────────┴────────┘

Saved to: nms_sweep_results/nms_sweep_summary.csv
```

#### 4. Analyze Results

View the CSV:
```bash
cat nms_sweep_results/nms_sweep_summary.csv
```

Or process programmatically:
```python
import pandas as pd
df = pd.read_csv('nms_sweep_results/nms_sweep_summary.csv')
print(df)
# Find best by each metric
print("Best mAP@50:", df.loc[df['mAP@50'].idxmax()])
```

## Complete Timeline

| Time | Action | Command |
|------|--------|---------|
| t=0s | Submit | `python run_nms_sweep.py ...` |
| t=10s | 19 jobs queued | `squeue -u $(whoami)` shows all jobs |
| t=1-2h | Jobs running | `watch squeue -u $(whoami)` |
| t=2-4h | Jobs complete | `squeue -u $(whoami)` shows 0 jobs |
| t=4h+ | Collect | `python run_nms_sweep.py --collect` |
| t=5s | View CSV | `cat nms_sweep_results/nms_sweep_summary.csv` |

## If Something Goes Wrong

**Job keeps queued (not running)?**
```bash
squeue -u $(whoami)  # Check state
# If all "CA" (cancelled) or stuck - check resource availability
```

**Job failed?**
```bash
# Check error log
cat nms_sweep_results/logs/12345.err

# Check stdout for error messages
tail -50 nms_sweep_results/logs/12345.out
```

**Need to cancel?**
```bash
# Cancel all NMS jobs
bash nms_monitor.sh cancel

# Or specific job
scancel 12345
```

**Re-run specific threshold?**
```bash
# Just resubmit
bash nms_sweep_results/job_0_500.sh
```

## Interpreting Results

From the example output:

| Metric | Interpretation |
|--------|-----------------|
| **NMS IoU** | IoU threshold used |
| **mAP@50** | Accuracy at IoU=0.50 |
| **mAP@50:95** | Average accuracy (IoU 0.50-0.95) |
| **Precision** | Of detected objects, % correct |
| **Recall** | Of actual objects, % detected |
| **F1** | Harmonic mean of Precision & Recall |

**Finding optimal:**
- Best overall = highest mAP@50:95
- Best precision = lower NMS threshold (fewer overlaps removed)
- Best recall = higher NMS threshold (more overlaps removed)

## Next Steps

1. **Identify best threshold** - Look at your priorities (accuracy vs precision vs recall)
2. **Use in production** - `dfine test -m model.ckpt -e test.txt --nms-iou 0.5 test_set/`
3. **Store result** - Save the optimal threshold in your config

## Files Created

```
nms_sweep_results/
├── job_none.sh              # 19 SLURM job scripts
├── job_0_100.sh
├── job_0_150.sh
├── ...
├── job_0_950.sh
├── logs/                    # Job output/errors (created by SLURM)
│   ├── 12345.out
│   ├── 12345.err
│   └── ...
├── results_nms_none.json    # Raw JSON results from each job
├── results_nms_0_100.json
├── ...
└── nms_sweep_summary.csv    # Final summary table
```

## Key SLURM Commands

```bash
# See all your jobs
squeue -u $(whoami)

# See NMS jobs
squeue -u $(whoami) | grep nms

# View job details
scontrol show job 12345

# Check why job is waiting
scontrol show job 12345 | grep Reason

# Cancel all NMS jobs
scancel -u $(whoami) --name="*nms*"

# View job history
sacct -u $(whoami) --format=JobID,JobName,State,Elapsed
```

## That's It!

The sweep runs all 19 thresholds in parallel, and SLURM handles all the job management. Just submit, monitor with `squeue`, and collect results when done.
