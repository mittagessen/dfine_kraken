#!/usr/bin/env python3
"""
Simple NMS sweep - submit parallel sbatch jobs for NMS IoU thresholds.
"""
import argparse
import subprocess
import json
from pathlib import Path


# Predefined thresholds (empirically chosen)
THRESHOLDS = [
    None,   # baseline, no NMS
    0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500,
    0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950
]


def create_sbatch_script(model: str, test_data: str, test_set: str, 
                        nms_iou, output_dir: str, job_name: str) -> str:
    """Create a single sbatch job script."""
    script_path = Path(output_dir) / f"job_{job_name}.sh"
    
    # Format NMS parameter
    nms_str = "none" if nms_iou is None else f"{nms_iou:.3f}"
    nms_param = "" if nms_iou is None else f"--nms-iou {nms_iou}"
    
    script = f"""#!/bin/bash
#SBATCH --job-name=nms_{nms_str}
#SBATCH --output={output_dir}/logs/%j.out
#SBATCH --error={output_dir}/logs/%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx8000:1

source /home/jjanes/Documents/4_LADaS/dfine_kraken/venv-test-dfine/bin/activate
cd /home/jjanes/Documents/4_LADaS/dfine_kraken

dfine test -m {model} -e {test_data} {nms_param} {test_set}
"""
    
    script_path.write_text(script)
    script_path.chmod(0o755)
    return str(script_path)


def submit_jobs(model: str, test_data: str, test_set: str, output_dir: str, dry_run: bool = False):
    """Submit all NMS sweep jobs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    print(f"Submitting {len(THRESHOLDS)} NMS sweep jobs...")
    print(f"Model: {model}")
    print(f"Test data: {test_data}")
    print(f"Test set: {test_set}\n")
    
    job_ids = []
    for i, nms_iou in enumerate(THRESHOLDS, 1):
        nms_str = "none" if nms_iou is None else f"{nms_iou:.3f}"
        script_path = create_sbatch_script(model, test_data, test_set, nms_iou, str(output_dir), nms_str)
        
        if dry_run:
            print(f"  [{i:2d}] Would submit: {script_path}")
        else:
            try:
                result = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True)
                job_id = result.stdout.split()[-1]
                job_ids.append(job_id)
                print(f"  [{i:2d}] NMS={nms_str:>7} → Job {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting {nms_str}: {e.stderr}")
                return
    
    if not dry_run:
        print(f"\n✓ Submitted {len(job_ids)} jobs")
        print(f"Check status: squeue -u $(whoami)")
        print(f"Results: {output_dir}/nms_sweep_summary.csv")


def collect_results(output_dir: str):
    """Collect and display results in a nice table."""
    output_dir = Path(output_dir)
    result_files = sorted(output_dir.glob("results_nms_*.json"))
    
    if not result_files:
        print(f"No results found in {output_dir}")
        return
    
    results = []
    for f in result_files:
        try:
            data = json.load(open(f))
            results.append({
                'NMS IoU': data.get('nms_iou', 'N/A'),
                'mAP@50': f"{float(data.get('map_50', 0)):.4f}",
                'mAP@50:95': f"{float(data.get('map_50_95', 0)):.4f}",
                'Precision': f"{float(data.get('precision', 0)):.4f}",
                'Recall': f"{float(data.get('recall', 0)):.4f}",
                'F1': f"{float(data.get('f1', 0)):.4f}"
            })
        except:
            pass
    
    if not results:
        print("No valid results to display")
        return
    
    # Sort: None first, then by threshold value
    def sort_key(x):
        iou = x['NMS IoU']
        if iou == 'N/A' or iou is None:
            return -1
        try:
            return float(iou) if iou != 'None' else -1
        except:
            return 999
    
    results.sort(key=sort_key)
    
    # Print table
    print("\n┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓")
    print("┃ NMS IoU ┃ mAP@50 ┃ mAP@50:95 ┃ Precision ┃ Recall ┃     F1 ┃")
    print("┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩")
    
    for row in results:
        iou_val = str(row['NMS IoU'])
        if iou_val == 'None' or iou_val == 'N/A':
            iou_str = "none"
        else:
            try:
                iou_str = f"{float(iou_val):.3f}"
            except:
                iou_str = iou_val
        
        print(f"│ {iou_str:>7} │ {row['mAP@50']:>6} │ {row['mAP@50:95']:>9} │ {row['Precision']:>9} │ {row['Recall']:>6} │ {row['F1']:>6} │")
    
    print("└─────────┴────────┴───────────┴───────────┴────────┴────────┘")
    
    # Save CSV
    csv_path = output_dir / "nms_sweep_summary.csv"
    with open(csv_path, 'w') as f:
        f.write(','.join(results[0].keys()) + '\n')
        for row in results:
            f.write(','.join(row.values()) + '\n')
    print(f"\nSaved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='NMS sweep - test multiple NMS thresholds in parallel')
    parser.add_argument('-m', '--model', help='Model path')
    parser.add_argument('-e', '--test-data', help='Test data manifest')
    parser.add_argument('test_set', nargs='?', help='Test set directory')
    parser.add_argument('-o', '--output-dir', default='nms_sweep_results', help='Output dir')
    parser.add_argument('--dry-run', action='store_true', help='Show jobs without submitting')
    parser.add_argument('--collect', action='store_true', help='Collect results')
    
    args = parser.parse_args()
    
    if args.collect:
        collect_results(args.output_dir)
    else:
        if not args.model or not args.test_data or not args.test_set:
            parser.error('Need: -m MODEL -e TEST_DATA TEST_SET (or --collect)')
        submit_jobs(args.model, args.test_data, args.test_set, args.output_dir, args.dry_run)


if __name__ == '__main__':
    main()
