#!/bin/bash
# Example: Run NMS sweep

# Step 1: Submit all 19 NMS sweep jobs
python run_nms_sweep.py \
    -m /path/to/model.ckpt \
    -e /path/to/test_manifest.txt \
    /path/to/test_set

# Step 2: Wait for jobs to complete (about 2-4 hours for 19 parallel jobs)
# Monitor with:
#   squeue -u $(whoami)
#   bash nms_monitor.sh
#   bash nms_monitor.sh tail

# Step 3: Collect results
python run_nms_sweep.py --collect

# Step 4: View results
cat nms_sweep_results/nms_sweep_summary.csv
