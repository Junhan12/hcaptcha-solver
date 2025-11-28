"""
Modular postprocessing functions for detection results.
Each operation is implemented as a separate function that can be chained together.
"""

from app.utils.logger import get_logger

# Initialize logger
log = get_logger("postprocess")


# =========================
# Individual postprocessing operations
# =========================

def apply_nms(detections, params):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
    
    Note: YOLO already applies NMS during inference, but this can be used
    for additional filtering with different thresholds if needed.
    
    params: {
        "iou_threshold": float (default: 0.5),
        "confidence_threshold": float (default: 0.7)
    }
    
    Args:
        detections: List of detection dictionaries
        params: Dictionary with NMS parameters
    
    Returns:
        List of detection dictionaries after NMS
    """
    if not detections:
        return detections
    
    iou_threshold = params.get("iou_threshold", 0.5)
    confidence_threshold = params.get("confidence_threshold", 0.7)
    
    # Filter by confidence first
    filtered = [
        det for det in detections
        if isinstance(det, dict) and det.get('confidence', 0) >= confidence_threshold
    ]
    
    if len(filtered) <= 1:
        return filtered
    
    # Sort by confidence (descending)
    filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Apply NMS: remove boxes with high IoU overlap
    keep = []
    while filtered:
        # Take the highest confidence detection
        current = filtered.pop(0)
        keep.append(current)
        
        # Remove all detections with high IoU overlap
        filtered = [
            det for det in filtered
            if _calculate_iou(current.get('bbox', []), det.get('bbox', [])) < iou_threshold
        ]
    
    if len(keep) < len(detections):
        log.info(f"NMS filtered {len(detections)} → {len(keep)} detections (iou_threshold: {iou_threshold}, conf_threshold: {confidence_threshold})", indent=1)
    
    return keep


def apply_aspect_ratio_filter(detections, params):
    """
    Reclassify square and diamond detections based on bounding box aspect ratio.
    
    Logic:
    - If aspect ratio < 0.9 OR > 1.1 → diamond (non-square shape)
    - If 0.9 <= aspect ratio <= 1.1 → square (approximately square)
    
    params: {} (no parameters needed, uses default thresholds)
    
    Args:
        detections: List of detection dictionaries
        params: Dictionary with operation parameters (currently unused)
    
    Returns:
        List of detection dictionaries with potentially reclassified classes
    """
    if not detections:
        return detections
    
    processed_detections = []
    reclassified_count = 0
    
    for det in detections:
        # Only process square and diamond detections
        if not isinstance(det, dict):
            processed_detections.append(det)
            continue
        
        class_name = det.get('class', '').lower()
        bbox = det.get('bbox', [])
        
        # Check if this is a square or diamond detection
        if class_name not in ['square', 'diamond']:
            # Not a square or diamond, keep as is
            processed_detections.append(det)
            continue
        
        # Validate bbox format: [x1, y1, x2, y2]
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            # Invalid bbox, keep as is
            processed_detections.append(det)
            continue
        
        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # Calculate width and height
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            
            # Avoid division by zero
            if height == 0:
                # Invalid height, keep as is
                processed_detections.append(det)
                continue
            
            # Calculate aspect ratio (width/height)
            aspect_ratio = width / height
            
            # Reclassify based on aspect ratio
            new_class = class_name  # Default: keep original class
            
            if aspect_ratio < 0.9 or aspect_ratio > 1.1:
                # Non-square shape → diamond
                new_class = 'diamond'
            elif 0.9 <= aspect_ratio <= 1.1:
                # Approximately square → square
                new_class = 'square'
            
            # Create new detection with potentially reclassified class
            new_det = det.copy()
            new_det['class'] = new_class
            
            # Log reclassification if class changed
            if new_class != class_name:
                log.info(f"Reclassified {class_name} → {new_class} (aspect_ratio: {aspect_ratio:.2f})", indent=1)
                reclassified_count += 1
            
            processed_detections.append(new_det)
            
        except (ValueError, TypeError) as e:
            # Error processing bbox, keep as is
            log.warning(f"Error processing bbox: {e}", indent=1)
            processed_detections.append(det)
            continue
    
    if reclassified_count > 0:
        log.info(f"Aspect ratio filter reclassified {reclassified_count} detection(s)", indent=1)
    
    return processed_detections


def _calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    try:
        x1_1, y1_1, x2_1, y2_1 = float(bbox1[0]), float(bbox1[1]), float(bbox1[2]), float(bbox1[3])
        x1_2, y1_2, x2_2, y2_2 = float(bbox2[0]), float(bbox2[1]), float(bbox2[2]), float(bbox2[3])
        
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
    except (ValueError, TypeError):
        return 0.0


# Operation registry
OPERATION_REGISTRY = {
    "nms": apply_nms,
    "aspect_ratio_filter": apply_aspect_ratio_filter,
}


# =========================
# Main postprocessing pipeline
# =========================

def apply_postprocess(detections, postprocess_profile):
    """
    Apply postprocessing steps from a postprocess_profile to detection results.
    
    Args:
        detections: List of detection dictionaries, each with:
            - 'class': class name (str)
            - 'confidence': confidence score (float)
            - 'bbox': [x1, y1, x2, y2] (list of floats)
        postprocess_profile: Dict with 'steps' array, each step has 'operation' and 'params'
    
    Returns:
        List of detection dictionaries after applying all postprocessing steps
    """
    if not postprocess_profile:
        return detections
    
    steps = postprocess_profile.get("steps", [])
    if not steps:
        return detections
    
    if not isinstance(steps, list):
        log.warning("Postprocess steps should be a list, got: {type(steps)}")
        return detections
    
    try:
        processed_detections = detections
        applied = []
        
        # Apply each step in sequence
        for step in steps:
            if not isinstance(step, dict):
                log.warning(f"Invalid step format (expected dict): {step}")
                continue
            
            operation_name = step.get("operation")
            params = step.get("params", {})
            
            if not operation_name:
                log.warning("Step missing 'operation' field, skipping")
                continue
            
            if operation_name not in OPERATION_REGISTRY:
                log.warning(f"Unknown operation '{operation_name}', skipping", indent=1)
                continue
            
            try:
                operation_func = OPERATION_REGISTRY[operation_name]
                processed_detections = operation_func(processed_detections, params)
                applied.append({"operation": operation_name, "params": params})
            except Exception as e:
                log.error(f"Error applying {operation_name}: {e}", indent=1)
                import traceback
                traceback.print_exc()
                # Continue with next step on error
                continue
        
        if applied:
            log.info(f"Applied {len(applied)} postprocessing step(s): {[s['operation'] for s in applied]}")
        
        return processed_detections
        
    except Exception as e:
        log.error(f"Postprocessing error: {e}")
        import traceback
        traceback.print_exc()
        # Return original detections on failure
        return detections
