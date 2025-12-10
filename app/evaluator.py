"""
YOLO Model Evaluation Metrics Calculator

This module provides functions to calculate evaluation metrics for YOLO object detection models:
- IoU (Intersection over Union) for bounding box matching
- Per-class Precision, Recall, F1-score
- Per-class AP (Average Precision) at IoU thresholds
- Overall mAP@0.5 and mAP@0.5:0.95
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import json
import yaml
import os
from pathlib import Path
from PIL import Image


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2] in pixel coordinates
        bbox2: [x1, y1, x2, y2] in pixel coordinates
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def yolo_to_xyxy(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert YOLO format (normalized center coordinates) to pixel coordinates (x1, y1, x2, y2).
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] normalized to [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert normalized to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to x1, y1, x2, y2
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    return [x1, y1, x2, y2]


def match_predictions_to_ground_truth(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    class_mapping: Optional[Dict[str, str]] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Match predictions to ground truth boxes using IoU threshold.
    
    Args:
        predictions: List of prediction dicts with 'class', 'confidence', 'bbox' [x1, y1, x2, y2]
        ground_truth: List of ground truth dicts with 'class', 'bbox' [x1, y1, x2, y2]
        iou_threshold: IoU threshold for matching (default 0.5)
        class_mapping: Optional dict to map prediction class names to ground truth class names
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
        Each is a list of dicts with 'class', 'bbox', 'confidence' (for predictions), 'iou' (for matches)
    """
    # Sort predictions by confidence (descending)
    sorted_predictions = sorted(predictions, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Group ground truth by class
    gt_by_class = defaultdict(list)
    for gt in ground_truth:
        class_name = gt.get('class', '')
        if class_mapping and class_name in class_mapping:
            class_name = class_mapping[class_name]
        gt_by_class[class_name].append(gt)
    
    true_positives = []
    false_positives = []
    matched_gt_indices = set()
    
    # Match each prediction to ground truth
    for pred in sorted_predictions:
        pred_class = pred.get('class', '')
        pred_bbox = pred.get('bbox', [])
        
        if len(pred_bbox) < 4:
            false_positives.append(pred)
            continue
        
        # Apply class mapping if provided
        if class_mapping and pred_class in class_mapping:
            pred_class = class_mapping[pred_class]
        
        # Find best matching ground truth of the same class
        best_iou = 0.0
        best_gt_idx = None
        best_gt = None
        
        if pred_class in gt_by_class:
            for idx, gt in enumerate(gt_by_class[pred_class]):
                gt_bbox = gt.get('bbox', [])
                if len(gt_bbox) < 4:
                    continue
                
                # Create unique identifier for this GT box
                gt_id = (pred_class, idx)
                
                if gt_id in matched_gt_indices:
                    continue
                
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_id
                    best_gt = gt
        
        # Check if match meets IoU threshold
        if best_iou >= iou_threshold and best_gt_idx is not None:
            matched_gt_indices.add(best_gt_idx)
            true_positives.append({
                'class': pred_class,
                'bbox': pred_bbox,
                'confidence': pred.get('confidence', 0.0),
                'iou': best_iou,
                'gt_bbox': best_gt.get('bbox', [])
            })
        else:
            false_positives.append(pred)
    
    # Find false negatives (unmatched ground truth)
    false_negatives = []
    for class_name, gt_list in gt_by_class.items():
        for idx, gt in enumerate(gt_list):
            gt_id = (class_name, idx)
            if gt_id not in matched_gt_indices:
                false_negatives.append(gt)
    
    return true_positives, false_positives, false_negatives


def calculate_precision_recall_f1(
    true_positives: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
    class_name: str
) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F1-score for a specific class.
    
    Args:
        true_positives: List of TP detections for this class
        false_positives: List of FP detections for this class
        false_negatives: List of FN detections for this class
        class_name: Class name to filter by
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    tp = len([tp for tp in true_positives if tp.get('class') == class_name])
    fp = len([fp for fp in false_positives if fp.get('class') == class_name])
    fn = len([fn for fn in false_negatives if fn.get('class') == class_name])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score


def calculate_ap(
    true_positives: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
    class_name: str,
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision (AP) for a specific class at a given IoU threshold.
    
    Uses the 11-point interpolation method (COCO style).
    
    Args:
        true_positives: List of TP detections
        false_positives: List of FP detections
        false_negatives: List of FN detections
        class_name: Class name to calculate AP for
        iou_threshold: IoU threshold used for matching
    
    Returns:
        AP value between 0 and 1
    """
    # Filter by class
    tp_class = [tp for tp in true_positives if tp.get('class') == class_name]
    fp_class = [fp for fp in false_positives if fp.get('class') == class_name]
    fn_class = [fn for fn in false_negatives if fn.get('class') == class_name]
    
    total_gt = len(tp_class) + len(fn_class)
    if total_gt == 0:
        return 0.0
    
    # Combine TP and FP, sort by confidence (descending)
    all_detections = []
    for tp in tp_class:
        all_detections.append({
            'confidence': tp.get('confidence', 0.0),
            'is_tp': True
        })
    for fp in fp_class:
        all_detections.append({
            'confidence': fp.get('confidence', 0.0),
            'is_tp': False
        })
    
    if len(all_detections) == 0:
        return 0.0
    
    # Sort by confidence descending
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Calculate cumulative TP and FP
    tp_cumsum = []
    fp_cumsum = []
    tp_count = 0
    fp_count = 0
    
    for det in all_detections:
        if det['is_tp']:
            tp_count += 1
        else:
            fp_count += 1
        tp_cumsum.append(tp_count)
        fp_cumsum.append(fp_count)
    
    # Calculate precision and recall at each threshold
    precisions = []
    recalls = []
    
    for i in range(len(all_detections)):
        tp_cum = tp_cumsum[i]
        fp_cum = fp_cumsum[i]
        precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0.0
        recall = tp_cum / total_gt if total_gt > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        # Find max precision where recall >= t
        max_precision = 0.0
        for i, rec in enumerate(recalls):
            if rec >= t:
                max_precision = max(max_precision, precisions[i])
        ap += max_precision / 11.0
    
    return ap


def calculate_map(
    true_positives: List[Dict],
    false_positives: List[Dict],
    false_negatives: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate mean Average Precision (mAP) across all classes at a given IoU threshold.
    
    Args:
        true_positives: List of TP detections
        false_positives: List of FP detections
        false_negatives: List of FN detections
        iou_threshold: IoU threshold used for matching
    
    Returns:
        mAP value between 0 and 1
    """
    # Get all unique classes
    all_classes = set()
    for tp in true_positives:
        all_classes.add(tp.get('class', ''))
    for fp in false_positives:
        all_classes.add(fp.get('class', ''))
    for fn in false_negatives:
        all_classes.add(fn.get('class', ''))
    
    all_classes.discard('')  # Remove empty class names
    
    if len(all_classes) == 0:
        return 0.0
    
    # Calculate AP for each class
    aps = []
    for class_name in all_classes:
        ap = calculate_ap(true_positives, false_positives, false_negatives, class_name, iou_threshold)
        aps.append(ap)
    
    # Return mean AP
    return np.mean(aps) if len(aps) > 0 else 0.0


def evaluate_model(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float = 0.5,
    class_mapping: Optional[Dict[str, str]] = None
) -> Dict:
    """
    Comprehensive evaluation of YOLO model predictions against ground truth.
    
    Args:
        predictions: List of prediction dicts with 'class', 'confidence', 'bbox' [x1, y1, x2, y2]
        ground_truth: List of ground truth dicts with 'class', 'bbox' [x1, y1, x2, y2]
        iou_threshold: IoU threshold for matching (default 0.5)
        class_mapping: Optional dict to map prediction class names to ground truth class names
    
    Returns:
        Dictionary containing:
        - 'per_class_metrics': Dict with class names as keys, containing precision, recall, f1, ap
        - 'overall_metrics': Dict with overall mAP@0.5, mAP@0.5:0.95, total_tp, total_fp, total_fn
        - 'true_positives': List of TP detections
        - 'false_positives': List of FP detections
        - 'false_negatives': List of FN detections
    """
    # Match predictions to ground truth
    tp, fp, fn = match_predictions_to_ground_truth(
        predictions, ground_truth, iou_threshold, class_mapping
    )
    
    # Get all unique classes
    all_classes = set()
    for item in tp + fp + fn + ground_truth:
        class_name = item.get('class', '')
        if class_mapping and class_name in class_mapping:
            class_name = class_mapping[class_name]
        if class_name:
            all_classes.add(class_name)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_name in all_classes:
        precision, recall, f1 = calculate_precision_recall_f1(tp, fp, fn, class_name)
        ap_50 = calculate_ap(tp, fp, fn, class_name, iou_threshold=0.5)
        
        per_class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'ap_50': ap_50,
            'tp': len([x for x in tp if x.get('class') == class_name]),
            'fp': len([x for x in fp if x.get('class') == class_name]),
            'fn': len([x for x in fn if x.get('class') == class_name])
        }
    
    # Calculate overall metrics
    map_50 = calculate_map(tp, fp, fn, iou_threshold=0.5)
    
    # Calculate mAP@0.5:0.95 (average of mAP at IoU thresholds from 0.5 to 0.95 in steps of 0.05)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = []
    for iou_thresh in iou_thresholds:
        tp_thresh, fp_thresh, fn_thresh = match_predictions_to_ground_truth(
            predictions, ground_truth, iou_thresh, class_mapping
        )
        map_val = calculate_map(tp_thresh, fp_thresh, fn_thresh, iou_thresh)
        maps.append(map_val)
    map_50_95 = np.mean(maps) if len(maps) > 0 else 0.0
    
    overall_metrics = {
        'map_50': map_50,
        'map_50_95': map_50_95,
        'total_tp': len(tp),
        'total_fp': len(fp),
        'total_fn': len(fn),
        'total_predictions': len(predictions),
        'total_ground_truth': len(ground_truth)
    }
    
    return {
        'per_class_metrics': per_class_metrics,
        'overall_metrics': overall_metrics,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def load_ground_truth_from_yolo_format(
    annotation_file: str,
    class_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Load ground truth annotations from YOLO format text file.
    
    YOLO format: class_id x_center y_center width height (all normalized to [0, 1])
    One line per object.
    
    Args:
        annotation_file: Path to YOLO format annotation file
        class_names: Optional list of class names (indexed by class_id)
    
    Returns:
        List of ground truth dicts with 'class' and 'bbox' [x1, y1, x2, y2] in pixel coordinates
    """
    ground_truth = []
    
    # Note: This function requires image dimensions to convert normalized coordinates
    # For now, we'll assume a default size or require it as a parameter
    # In practice, you'd get this from the image file
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f'class_{class_id}'
            
            # Store in normalized format - will need image dimensions to convert
            ground_truth.append({
                'class': class_name,
                'bbox_yolo': [x_center, y_center, width, height],
                'class_id': class_id
            })
    
    return ground_truth


def load_ground_truth_from_json(json_file: str) -> List[Dict]:
    """
    Load ground truth annotations from JSON file.
    
    Expected JSON format:
    [
        {
            "class": "class_name",
            "bbox": [x1, y1, x2, y2]  // in pixel coordinates
        },
        ...
    ]
    
    Args:
        json_file: Path to JSON annotation file
    
    Returns:
        List of ground truth dicts
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'annotations' in data:
        return data['annotations']
    else:
        raise ValueError("Invalid JSON format. Expected list of annotations or dict with 'annotations' key.")


def parse_data_yaml(yaml_file: str) -> Dict:
    """
    Parse Roboflow-style data.yaml file.
    
    Expected YAML format:
    path: ../datasets/dataset_name
    train: images/train
    val: images/val
    test: images/test
    names:
      0: class1
      1: class2
    nc: 2
    
    Args:
        yaml_file: Path to data.yaml file
    
    Returns:
        Dictionary with parsed YAML data
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def load_yolo_annotation(annotation_file: str, img_width: int, img_height: int, class_names: Optional[Dict[int, str]] = None) -> List[Dict]:
    """
    Load YOLO format annotation file and convert to pixel coordinates.
    
    YOLO format: class_id x_center y_center width height (all normalized to [0, 1])
    One line per object.
    
    Args:
        annotation_file: Path to YOLO format annotation file (.txt)
        img_width: Image width in pixels
        img_height: Image height in pixels
        class_names: Optional dict mapping class_id to class name
    
    Returns:
        List of ground truth dicts with 'class' and 'bbox' [x1, y1, x2, y2] in pixel coordinates
    """
    ground_truth = []
    
    if not os.path.exists(annotation_file):
        return ground_truth
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Get class name
            if class_names and class_id in class_names:
                class_name = class_names[class_id]
            else:
                class_name = f'class_{class_id}'
            
            # Convert YOLO format (normalized center) to pixel coordinates (x1, y1, x2, y2)
            bbox = yolo_to_xyxy([x_center, y_center, width, height], img_width, img_height)
            
            ground_truth.append({
                'class': class_name,
                'bbox': bbox,
                'class_id': class_id
            })
    
    return ground_truth


def load_validation_dataset(data_yaml_path: str) -> Tuple[List[str], List[List[Dict]], Dict[int, str]]:
    """
    Load validation dataset from Roboflow-style data.yaml file.
    
    NOTE: This function ONLY READS files. It does NOT modify, write, or convert
    any files in the dataset. Images are opened in read-only mode.
    
    Args:
        data_yaml_path: Path to data.yaml file
    
    Returns:
        Tuple of:
        - List of image file paths (validation set)
        - List of ground truth annotations (one list per image)
        - Dict mapping class_id to class name
    """
    # Parse data.yaml
    yaml_data = parse_data_yaml(data_yaml_path)
    
    # Get base path
    base_path = yaml_data.get('path', '')
    if not base_path:
        # If path is relative, use directory of yaml file
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        base_path = yaml_dir
    else:
        # Resolve relative path from yaml file location
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        if not os.path.isabs(base_path):
            base_path = os.path.join(yaml_dir, base_path)
        base_path = os.path.abspath(base_path)
    
    # Get validation split path
    val_path = yaml_data.get('val', '')
    if not val_path:
        raise ValueError("No 'val' key found in data.yaml")
    
    # Resolve validation directory
    val_dir = os.path.join(base_path, val_path) if not os.path.isabs(val_path) else val_path
    val_dir = os.path.abspath(val_dir)
    
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory not found: {val_dir}")
    
    # Get class names
    names = yaml_data.get('names', {})
    class_names = {}
    if isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        class_names = {i: str(name) for i, name in enumerate(names)}
    
    # Find images directory (usually 'images' subdirectory in val)
    images_dir = os.path.join(val_dir, 'images')
    if not os.path.exists(images_dir):
        # Try val_dir itself (val might already point to images directory)
        images_dir = val_dir
    
    # Find labels directory
    # Handle case where val_path points directly to 'images' directory
    # In that case, labels should be in sibling 'labels' directory
    if os.path.basename(val_dir).lower() == 'images':
        # val_dir is already the images directory, so labels should be in parent/labels
        parent_dir = os.path.dirname(val_dir)
        labels_dir = os.path.join(parent_dir, 'labels')
        if not os.path.exists(labels_dir):
            # Fallback: try val_dir itself (though this is unlikely to have labels)
            labels_dir = val_dir
    else:
        # val_dir is the parent directory, labels should be in val_dir/labels
        labels_dir = os.path.join(val_dir, 'labels')
        if not os.path.exists(labels_dir):
            # Try val_dir itself
            labels_dir = val_dir
    
    # Get all image files
    # Use a set to automatically deduplicate paths (handles case-insensitive filesystems)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files_set = set()
    
    for ext in image_extensions:
        # Collect files with lowercase extension
        for file_path in Path(images_dir).glob(f'*{ext}'):
            # Use resolve() to normalize path (handles case-insensitive filesystems like Windows)
            image_files_set.add(file_path.resolve())
        
        # Collect files with uppercase extension (for case-sensitive filesystems)
        for file_path in Path(images_dir).glob(f'*{ext.upper()}'):
            image_files_set.add(file_path.resolve())
    
    # Convert to sorted list of strings
    image_files = sorted([str(f) for f in image_files_set])
    
    # Load annotations for each image
    all_annotations = []
    valid_image_files = []
    
    # Debug: Print directory paths
    print(f"Debug: images_dir = {images_dir}")
    print(f"Debug: labels_dir = {labels_dir}")
    print(f"Debug: labels_dir exists = {os.path.exists(labels_dir)}")
    
    for img_path in image_files:
        # Get corresponding annotation file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        annotation_file = os.path.join(labels_dir, f"{img_name}.txt")
        
        # Load image to get dimensions (READ-ONLY - does not modify the file)
        try:
            # Open image in read-only mode - this does NOT modify or convert the file
            with Image.open(img_path) as img:
                img_width, img_height = img.size
                # Ensure we're not modifying anything - just reading dimensions
                img.load()  # Load image data into memory (still read-only)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue
        
        # Load annotations
        annotations = load_yolo_annotation(annotation_file, img_width, img_height, class_names)
        if annotations:
            print(f"Debug: Loaded {len(annotations)} annotations from {annotation_file}")
        else:
            print(f"Debug: No annotations found in {annotation_file} (file exists: {os.path.exists(annotation_file)})")
        all_annotations.append(annotations)
        valid_image_files.append(img_path)
    
    return valid_image_files, all_annotations, class_names

