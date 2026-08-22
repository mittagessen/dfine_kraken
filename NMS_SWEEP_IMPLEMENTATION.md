# NMS Parallel Sweep Implementation Summary

This document summarizes the NMS parallel sweep functionality created for testing multiple NMS IoU thresholds in parallel using sbatch.

## Summary

The NMS sweep system allows you to easily test multiple NMS IoU thresholds (0.0, 0.05, 0.10, ..., 0.92) in parallel:

1. **Simple setup**: Just provide model path and test data
2. **Automatic parallelization**: All 19 thresholds run simultaneously on the cluster
3. **Built on CLI**: Each job runs `dfine test -m model.ckpt -e test.txt --nms-iou <value> test_set`
4. **Comprehensive results**: Automatically collects mAP, Precision, Recall, F1 metrics
5. **Easy analysis**: Summary table and CSV for comparing results across thresholds

## Files Created

### 1. `run_nms_sweep.py` - Main sweep script
Main orchestration script that:
- Generates NMS IoU thresholds (default: 0.0, 0.05, 0.10, ..., 0.90, 0.92)
- Creates sbatch scripts for each threshold
- Submits all jobs to the cluster in parallel
- Collects and aggregates results

**Key Functions:**
- `generate_nms_thresholds()`: Generate sweep thresholds
- `create_sbatch_script()`: Create individual sbatch job scripts
- `submit_jobs()`: Submit all jobs to sbatch
- `collect_results()`: Collect and display summary table

**Usage:**
```bash
# Submit jobs
python run_nms_sweep.py -m model.ckpt -e test.txt test_dir

# Collect results
python run_nms_sweep.py --collect
```

### 2. `dfine/nms_utils.py` - Overlap analysis utilities
Helper module for computing overlapping detections:

**Functions:**
- `compute_iou()`: Calculate intersection-over-union between two boxes
- `count_overlapping_predictions()`: Count all overlapping detection pairs
- `count_same_class_overlaps()`: Count overlaps of the same class
- `count_duplicate_detections()`: Count potential duplicate detections (high IoU)

**Usage:**
```python
from dfine.nms_utils import count_overlapping_predictions
overlaps = count_overlapping_predictions(predictions)
```

### 3. `NMS_SWEEP_GUIDE.md` - User guide
Comprehensive guide covering:
- Overview of the sweep system
- Prerequisites and setup
- Basic and advanced usage examples
- Output file structure
- Result interpretation
- Troubleshooting

## Features

### Parallel Job Submission
- All NMS tests run in parallel on the cluster
- Default 19 thresholds = 19 parallel jobs
- Each job runs independently with different NMS IoU

### Comprehensive Metrics
Each job produces:
- **mAP@50**: Mean Average Precision at IoU=0.50
- **mAP@50:95**: Mean Average Precision at multiple IoU thresholds
- **Precision**: Detection precision
- **Recall**: Detection recall
- **F1**: Harmonic mean of precision and recall
- **Overlapping Zones**: Count of overlapping detections

### Summary Generation
Results are aggregated into:
- **HTML/Terminal Table**: Easy-to-read formatted table
- **CSV File**: For further analysis
- **Best Configurations**: Highlighted best performers

### Flexible Configuration
Options to customize:
- NMS IoU threshold range (--start, --end, --step)
- Overlap counting method (--overlap-type)
- Output directory (-o, --output-dir)
- Dry run mode (--dry-run)

## Workflow Example

```bash
# 1. Submit 19 parallel NMS sweep jobs
python run_nms_sweep.py \
    -m models/model_checkpoint.ckpt \
    -e data/test_manifest.txt \
    data/test_images \
    -o nms_results

# 2. Monitor progress
watch squeue -u $(whoami)
tail -f nms_results/logs/*.out

# 3. Once complete (2-4 hours), collect results
python run_nms_sweep.py --collect -o nms_results

# 4. View results
cat nms_results/nms_sweep_summary.csv
```

## Output Example

```
NMS IoU | Overlaps | mAP@50 | mAP@50:95 | Precision | Recall | F1
--------|----------|--------|-----------|-----------|--------|-----
0.00    | 245      | 0.623  | 0.542     | 0.812     | 0.734  | 0.771
0.05    | 232      | 0.625  | 0.544     | 0.814     | 0.736  | 0.773
0.10    | 218      | 0.627  | 0.546     | 0.816     | 0.738  | 0.775
...     | ...      | ...    | ...       | ...       | ...    | ...
0.92    | 124      | 0.631  | 0.548     | 0.819     | 0.740  | 0.777

Best Configurations:
- Best mAP@50:95: NMS IoU=0.35, mAP=0.552
- Best Precision: NMS IoU=0.40, Precision=0.823
- Best Recall: NMS IoU=0.15, Recall=0.745
- Fewest overlaps: NMS IoU=0.92, Overlaps=124
```

## Technical Details

### SLURM Configuration
Each sbatch job has:
- 2-hour time limit
- 32GB GPU memory
- Single GPU (--gpus=1)
- Auto-generated job logs

### Environment Setup
- Uses project's venv: `venv-test-dfine/bin/activate`
- Sets float32 matmul precision for torch
- Enables GPU acceleration

### Results Structure
```
nms_sweep_results/
├── logs/
│   ├── <job_id>.out
│   └── <job_id>.err
├── sbatch_nms_*.sh          # Sbatch scripts
├── results_nms_*.json       # Individual results
└── nms_sweep_summary.csv    # Aggregated summary
```

## NMS IoU Interpretation

- **0.0**: No NMS applied (baseline - most overlaps)
- **0.05-0.20**: Light filtering (minimal overlap removal)
- **0.30-0.50**: Moderate filtering (good balance)
- **0.60-0.80**: Heavy filtering (most overlaps removed)
- **0.90+**: Very aggressive (maximum overlap removal)

## Performance Expectations

- **Each job**: 1-2 hours runtime
- **Total time**: ~2-4 hours (parallel execution)
- **Memory**: ~32GB per job
- **Results**: 19 different model evaluations in one batch

## Future Enhancements

Possible improvements:
1. Adaptive NMS (different thresholds per class)
2. Soft-NMS variants
3. Per-image overlap statistics
4. Confidence-score analysis
5. Automatic optimal threshold selection

## Integration Points

### With existing DFINE code:
- Uses `DFINESegmentationModel` for predictions
- Uses `DFINESegmentationTrainingConfig` for model config
- Uses `DFINESegmentationTestDataConfig` for data config
- Integrates with `KrakenTrainer` for inference

### With K raken framework:
- Uses `KrakenTrainer` for model evaluation
- Compatible with existing checkpoint format
- Uses standard test metrics computation

## Notes

- NMS is applied during model forward pass (no post-hoc filtering)
- Overlapping count is computed on raw predictions before NMS
- Results are JSON for easy parsing and integration
- CSV output for analysis in spreadsheet tools
- All jobs are independent and can be cancelled without affecting others
