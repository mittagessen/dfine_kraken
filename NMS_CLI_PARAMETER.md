# DFINE Test with NMS IoU Parameter

The `dfine test` command now supports the `--nms-iou` parameter to control Non-Maximum Suppression during inference.

## Basic Usage

### No NMS (default)
```bash
dfine test -m model.ckpt -e test_manifest.txt test_set/
```

### With NMS
```bash
dfine test -m model.ckpt -e test_manifest.txt --nms-iou 0.5 test_set/
```

## Parameter Details

- **`--nms-iou`**: Float value between 0.0 and 1.0
- **Default**: `None` (no NMS applied)
- **Effect**: Removes overlapping detections with IoU >= threshold

## Examples

### Very light NMS (only remove very close duplicates)
```bash
dfine test -m model.ckpt -e test.txt --nms-iou 0.1 test_set/
```

### Moderate NMS (recommended for most cases)
```bash
dfine test -m model.ckpt -e test.txt --nms-iou 0.5 test_set/
```

### Aggressive NMS (remove most overlaps)
```bash
dfine test -m model.ckpt -e test.txt --nms-iou 0.9 test_set/
```

## What NMS does

NMS (Non-Maximum Suppression) removes overlapping bounding boxes:

1. Sorts detections by confidence score (highest first)
2. Keeps the highest confidence detection
3. Removes any other detections with IoU >= threshold
4. Repeats for remaining detections

**Effect of different thresholds:**

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| 0.0 | No NMS applied | Baseline, see all detections |
| 0.1-0.3 | Light filtering | Minimal overlap removal |
| 0.4-0.6 | Moderate filtering | Good balance, typical use |
| 0.7-0.9 | Heavy filtering | Remove most overlaps |
| 0.95+ | Very aggressive | Only keep one detection per region |

## Output Format

The test output will show:

```
Overall Detection Metrics
┌────────────┬──────────────┬───────────┬────────┬────────┐
│ mAP@50     │ mAP@50:95    │ Precision │ Recall │ F1     │
├────────────┼──────────────┼───────────┼────────┼────────┤
│ 0.6234     │ 0.5421       │ 0.8123    │ 0.7345 │ 0.7702 │
└────────────┴──────────────┴───────────┴────────┴────────┘
```

The metrics shown already include the effect of NMS with your chosen threshold.

## Using in a Script

### Python
```python
from click.testing import CliRunner
from dfine.cli.test import test

runner = CliRunner()
result = runner.invoke(test, [
    '-m', 'model.ckpt',
    '-e', 'test.txt',
    '--nms-iou', '0.5',
    'test_set/'
])
print(result.output)
```

### Bash
```bash
#!/bin/bash
for nms_iou in 0.0 0.1 0.2 0.3 0.4 0.5; do
    echo "Testing with NMS IoU = $nms_iou"
    dfine test -m model.ckpt -e test.txt --nms-iou $nms_iou test_set/
    echo "---"
done
```

## Parallel Testing (Recommended)

For testing multiple NMS thresholds in parallel, use the sweep script:

```bash
# Test all thresholds from 0.0 to 0.92 in parallel
python run_nms_sweep.py -m model.ckpt -e test.txt test_set/

# Collect results
python run_nms_sweep.py --collect
```

See `NMS_SWEEP_QUICKSTART.md` for details.

## Tips

1. **Finding optimal NMS**: Run the sweep script to test multiple thresholds
2. **Balancing metrics**: Lower NMS threshold = more detections (higher recall) but potentially more false positives
3. **Typical range**: Most models work well with NMS IoU between 0.3-0.6
4. **None vs 0.0**: If NMS threshold is 0.0, no NMS is applied (same as None)

## Integration with Training

The NMS threshold can also be set during training in the config:

```python
config = DFINESegmentationTrainingDataConfig(
    nms_iou=0.5  # Set NMS threshold
)
```

But for evaluation/testing, it's typically set via the CLI `--nms-iou` flag.

## Comparison: With vs Without NMS

Example scenario:
- Model detects 1000 total boxes
- Without NMS: all 1000 boxes evaluated
- With NMS (0.5): ~600 boxes remain after filtering overlaps
- With NMS (0.9): ~800 boxes remain (lighter filtering)

This affects both speed and metrics.
