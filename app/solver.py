import io
import numpy as np
from PIL import Image
import tempfile
import os

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Import database function to get model weights
try:
    from .database import download_weights_bytes
except Exception:
    # Fallback for absolute import
    try:
        from app.database import download_weights_bytes
    except Exception:
        download_weights_bytes = None

# Model cache to avoid reloading models on every request
_model_cache = {}
_weights_cache = {}  # Cache weights bytes to avoid re-downloading
_temp_file_cache = {}  # Cache temp file paths for database-loaded models


def clear_model_cache(model_id=None):
    """
    Clear model cache. If model_id is provided, clears only that model's cache.
    If model_id is None, clears all cached models.
    
    Args:
        model_id: Optional model ID to clear. If None, clears all models.
    
    Returns:
        dict with information about what was cleared
    """
    cleared = {
        'models': 0,
        'weights': 0,
        'temp_files': 0,
        'temp_file_paths': []
    }
    
    if model_id:
        # Clear specific model
        if model_id in _model_cache:
            del _model_cache[model_id]
            cleared['models'] = 1
        
        if model_id in _weights_cache:
            del _weights_cache[model_id]
            cleared['weights'] = 1
        
        if model_id in _temp_file_cache:
            tmp_path = _temp_file_cache[model_id]
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    cleared['temp_files'] = 1
                    cleared['temp_file_paths'].append(tmp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {tmp_path}: {e}")
            del _temp_file_cache[model_id]
    else:
        # Clear all models
        cleared['models'] = len(_model_cache)
        cleared['weights'] = len(_weights_cache)
        _model_cache.clear()
        _weights_cache.clear()
        
        # Delete all temp files
        for model_id_key, tmp_path in list(_temp_file_cache.items()):
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    cleared['temp_files'] += 1
                    cleared['temp_file_paths'].append(tmp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file {tmp_path}: {e}")
        _temp_file_cache.clear()
    
    return cleared


def get_cache_info():
    """
    Get information about the current model cache.
    
    Returns:
        dict with cache statistics
    """
    return {
        'cached_models': list(_model_cache.keys()),
        'cached_weights': list(_weights_cache.keys()),
        'cached_temp_files': {k: v for k, v in _temp_file_cache.items()},
        'model_count': len(_model_cache),
        'weights_count': len(_weights_cache),
        'temp_file_count': len(_temp_file_cache)
    }


def solve_captcha(image, question, config, postprocess_profile=None, use_native_eval=False, imgsz=None):
    """
    Run YOLO model inference on image with optional postprocessing steps.
    
    Args:
        image: Image bytes, image array, or file path (str) if use_native_eval=True
        question: Question string for context
        config: Model configuration dict with 'model_id'
        postprocess_profile: Dict with 'steps' array (list of operations) or legacy dict with thresholds
        use_native_eval: If True, use YOLOv8 native evaluation mode (skip preprocessing, use file paths directly)
        imgsz: Image size for inference (int or tuple). If None, uses model default.
    
    Returns:
        List of detection results, each with 'class', 'confidence', 'bbox' [x1, y1, x2, y2]
        Or dict with 'error' key if an error occurred
    """
    if not YOLO_AVAILABLE:
        return {'error': 'ultralytics library not available. Install with: pip install ultralytics'}
    
    # Extract postprocess thresholds for YOLO inference
    # Support both new format (steps array) and legacy format (direct thresholds dict)
    # Default thresholds match YOLOv8's standard evaluation settings
    conf_threshold = 0.25  # Default confidence threshold
    iou_threshold = 0.45  # Default IoU threshold for NMS (matches YOLOv8 validation default)
    
    if postprocess_profile:
        # Check if it's the new format (with 'steps' array)
        if isinstance(postprocess_profile, dict) and 'steps' in postprocess_profile:
            steps = postprocess_profile.get('steps', [])
            if isinstance(steps, list):
                # Look for 'nms' operation to extract thresholds
                for step in steps:
                    if isinstance(step, dict) and step.get('operation') == 'nms':
                        params = step.get('params', {})
                        conf_threshold = params.get('confidence_threshold', conf_threshold)
                        iou_threshold = params.get('iou_threshold', iou_threshold)
                        break
        # Legacy format: direct thresholds dict
        elif isinstance(postprocess_profile, dict):
            conf_threshold = postprocess_profile.get('confidence_threshold', conf_threshold)
            iou_threshold = postprocess_profile.get('iou_threshold', iou_threshold)
   
    try:
       
        # Get model_id from config
        model_id = config.get('model_id')
        if not model_id:
            return {'error': 'no model selected'}
        
        # Check if model is already cached
        model = _model_cache.get(model_id)
        
        if model is None:
            # Model not in cache, need to load it
            print(f"Loading model {model_id} (not in cache)...")
            weights_bytes = None
            
            # Check if weights are cached
            if model_id in _weights_cache:
                weights_bytes = _weights_cache[model_id]
                print(f"Using cached weights for {model_id}")
            elif download_weights_bytes:
                # Try to load weights from database
                try:
                    print("start downloading weights from database")
                    weights_bytes = download_weights_bytes(model_id)
                    print("finished downloading weights from database")
                    if weights_bytes:
                        # Cache the weights bytes
                        _weights_cache[model_id] = weights_bytes
                        print(f"Downloaded and cached weights for {model_id}")
                except Exception as e:
                    print(f"Failed to load weights from database: {e}")
            
            if weights_bytes:
                # Check if temp file already exists for this model
                tmp_path = _temp_file_cache.get(model_id)
                
                if tmp_path is None or not os.path.exists(tmp_path):
                    # Create temporary file and save weights
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                        tmp_file.write(weights_bytes)
                        tmp_path = tmp_file.name
                    _temp_file_cache[model_id] = tmp_path
                    print(f"Created temp file for {model_id}: {tmp_path}")
                else:
                    print(f"Reusing existing temp file for {model_id}: {tmp_path}")
                
                try:
                    model = YOLO(tmp_path)
                    # Cache the loaded model
                    _model_cache[model_id] = model
                    print(f"Model {model_id} loaded and cached successfully")
                except Exception as e:
                    return {'error': f'Failed to load model from database: {str(e)}'}
            else:
                # Fallback: try to load by model_id (e.g., 'yolov8n', 'yolov8s', etc.)
                try:
                    model = YOLO(f'{model_id}.pt')
                    # Cache the loaded model
                    _model_cache[model_id] = model
                    print(f"Model {model_id} loaded from file and cached")
                except Exception as e:
                    return {'error': f'Failed to load model {model_id}: {str(e)}'}
        else:
            print(f"Using cached model {model_id}")
        
        if model is None:
            return {'error': 'no model selected'}
        
        # Build predict parameters matching YOLOv8 validation defaults
        predict_params = {
            'conf': conf_threshold,  # Confidence threshold
            'iou': iou_threshold,    # IoU threshold for NMS
            'verbose': False,
            'augment': False,        # No test-time augmentation for evaluation
        }
        
        # Add imgsz if specified (should match training size for consistency)
        if imgsz is not None:
            predict_params['imgsz'] = imgsz
        
        # Handle image input based on evaluation mode
        if use_native_eval and isinstance(image, str) and os.path.exists(image):
            # Native evaluation mode: pass file path directly to YOLOv8
            # This matches YOLOv8's native evaluation behavior exactly
            image_source = image
        elif isinstance(image, bytes):
            # Image bytes - convert through PIL (for preprocessed images)
            img = Image.open(io.BytesIO(image))
            # Convert grayscale to RGB if needed
            if img.mode == 'L' or img.mode == 'LA' or img.mode == 'P':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                # Convert RGBA to RGB by creating a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            img_array = np.array(img)
            
            # Ensure image has 3 channels (RGB) for YOLO
            if len(img_array.shape) == 2:
                # Grayscale image (H, W) -> convert to RGB (H, W, 3)
                img_array = np.stack([img_array] * 3, axis=-1)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                # Grayscale with channel dimension (H, W, 1) -> convert to RGB (H, W, 3)
                img_array = np.repeat(img_array, 3, axis=2)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # RGBA image -> convert to RGB
                # Create white background and composite
                rgb_array = img_array[:, :, :3]
                alpha = img_array[:, :, 3:4] / 255.0
                white_bg = np.ones_like(rgb_array) * 255
                img_array = (rgb_array * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            
            image_source = img_array
        else:
            # Numpy array or other format
            image_source = image
        
        # Run inference with confidence and IoU thresholds
        # YOLO's predict method applies confidence filtering and NMS automatically
        results = model.predict(image_source, **predict_params)
        
        # Extract detections from YOLO results
        detections = []
        if results and len(results) > 0:
            result = results[0]  # Get first (and typically only) result
            
            # Get boxes, scores, and class IDs
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Get box coordinates (x1, y1, x2, y2)
                    box = boxes.xyxy[i].cpu().numpy()  # Convert to numpy array
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    
                    # Get confidence score
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class ID and name
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[class_id] if hasattr(result, 'names') else f'class_{class_id}'
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        # Apply postprocessing steps if provided (skip for native evaluation to match YOLOv8 exactly)
        if postprocess_profile and not use_native_eval:
            try:
                from .postprocess import apply_postprocess
            except ImportError:
                try:
                    from app.postprocess import apply_postprocess
                except ImportError:
                    apply_postprocess = None
            
            # Apply modular postprocessing steps
            if apply_postprocess:
                detections = apply_postprocess(detections, postprocess_profile)
        
        # Return empty list with message if no detections, or return detections
        if len(detections) == 0:
            return {'message': 'no detected output', 'detections': []}
        
        return detections
        
    except Exception as e:
        error_msg = str(e)
        print(f"YOLO inference error: {error_msg}")
        import traceback
        traceback.print_exc()
        return {'error': error_msg}

