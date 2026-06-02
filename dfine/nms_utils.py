#!/usr/bin/env python3
"""
Utility module for computing overlapping zones from detection results.
"""
from typing import List, Dict, Tuple
import torch


def compute_iou(box1: Tuple[float, float, float, float], 
                box2: Tuple[float, float, float, float]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: (x1, y1, x2, y2) format
        box2: (x1, y1, x2, y2) format
    
    Returns:
        IoU value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def count_overlapping_predictions(predictions: List[Dict]) -> int:
    """
    Count number of overlapping detection pairs (from same image).
    
    Args:
        predictions: List of prediction dicts with 'boxes' and 'labels'
                    Each dict should have:
                    - 'boxes': tensor of shape (N, 4) in xyxy format
                    - 'labels': tensor of shape (N,)
    
    Returns:
        Number of overlapping zone pairs
    """
    total_overlaps = 0
    
    for pred in predictions:
        if 'boxes' not in pred or len(pred['boxes']) == 0:
            continue
        
        boxes = pred['boxes']
        n_boxes = len(boxes)
        
        # Count pairs of boxes that overlap (IoU > 0)
        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                box1 = tuple(boxes[i].tolist()) if torch.is_tensor(boxes[i]) else boxes[i]
                box2 = tuple(boxes[j].tolist()) if torch.is_tensor(boxes[j]) else boxes[j]
                iou = compute_iou(box1, box2)
                if iou > 0:
                    total_overlaps += 1
    
    return total_overlaps


def count_same_class_overlaps(predictions: List[Dict], iou_threshold: float = 0.0) -> int:
    """
    Count overlapping detections with the same class label.
    
    Args:
        predictions: List of prediction dicts with 'boxes' and 'labels'
        iou_threshold: Minimum IoU to count as overlap (default 0.0)
    
    Returns:
        Number of overlapping same-class pairs
    """
    total_overlaps = 0
    
    for pred in predictions:
        if 'boxes' not in pred or 'labels' not in pred:
            continue
        if len(pred['boxes']) == 0:
            continue
        
        boxes = pred['boxes']
        labels = pred['labels']
        n_boxes = len(boxes)
        
        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                # Only count if same class
                if labels[i] != labels[j]:
                    continue
                
                box1 = tuple(boxes[i].tolist()) if torch.is_tensor(boxes[i]) else boxes[i]
                box2 = tuple(boxes[j].tolist()) if torch.is_tensor(boxes[j]) else boxes[j]
                iou = compute_iou(box1, box2)
                
                if iou > iou_threshold:
                    total_overlaps += 1
    
    return total_overlaps


def count_duplicate_detections(predictions: List[Dict], iou_threshold: float = 0.5) -> int:
    """
    Count detections that are duplicates (high IoU with same class).
    This is useful for understanding what NMS would have filtered.
    
    Args:
        predictions: List of prediction dicts
        iou_threshold: IoU threshold for considering as duplicate (default 0.5)
    
    Returns:
        Number of duplicate detection pairs
    """
    return count_same_class_overlaps(predictions, iou_threshold=iou_threshold)


if __name__ == '__main__':
    # Test example
    sample_pred = {
        'boxes': torch.tensor([[10, 10, 50, 50], [45, 45, 80, 80], [100, 100, 150, 150]]),
        'labels': torch.tensor([0, 0, 1])
    }
    
    overlaps = count_overlapping_predictions([sample_pred])
    print(f"Total overlaps: {overlaps}")
    
    same_class = count_same_class_overlaps([sample_pred])
    print(f"Same-class overlaps: {same_class}")
    
    duplicates = count_duplicate_detections([sample_pred], iou_threshold=0.3)
    print(f"Duplicate detections (IoU > 0.3): {duplicates}")
