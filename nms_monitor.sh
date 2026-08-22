#!/bin/bash
# Simple NMS monitoring using SLURM

if [ "$1" = "status" ] || [ -z "$1" ]; then
    echo "Current NMS jobs:"
    squeue -u $(whoami) | grep nms
    
elif [ "$1" = "logs" ]; then
    echo "Recent job logs:"
    ls -lht nms_sweep_results/logs/*.out 2>/dev/null | head -5
    
elif [ "$1" = "cancel" ]; then
    echo "Cancelling all NMS jobs..."
    scancel -u $(whoami) --name="*nms*"
    
elif [ "$1" = "tail" ]; then
    echo "Last job output:"
    tail -50 $(ls -t nms_sweep_results/logs/*.out 2>/dev/null | head -1)
    
else
    echo "Usage: bash nms_monitor.sh [status|logs|cancel|tail]"
    echo "  status - show running NMS jobs (default)"
    echo "  logs   - list recent log files"
    echo "  cancel - cancel all NMS jobs"
    echo "  tail   - show last job output"
fi
