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


def solve_captcha(image, question, config, postprocess_steps=None):
    """
    Run YOLO model inference on image with optional postprocessing steps.
    
    Args:
        image: Image bytes or image array
        question: Question string for context
        config: Model configuration dict with 'model_id'
        postprocess_steps: Dict with 'confidence_threshold' and 'iou_threshold' for NMS
    
    Returns:
        List of detection results, each with 'class', 'confidence', 'bbox' [x1, y1, x2, y2]
        Or dict with 'error' key if an error occurred
    """
    if not YOLO_AVAILABLE:
        return {'error': 'ultralytics library not available. Install with: pip install ultralytics'}
    
    # Extract postprocess thresholds if provided
    conf_threshold = 0.25  # Default
    iou_threshold = 0.45   # Default
    
    if postprocess_steps:
        conf_threshold = postprocess_steps.get('confidence_threshold', conf_threshold)
        iou_threshold = postprocess_steps.get('iou_threshold', iou_threshold)
    print("start solving captcha")
    try:
        print("in solver.py")
        
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
        
        # Convert image bytes to PIL Image or numpy array
        if isinstance(image, bytes):
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
        else:
            img_array = image
        
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
        
        # Run inference with confidence and IoU thresholds
        # YOLO's predict method applies confidence filtering and NMS automatically
        results = model.predict(
            img_array,
            conf=conf_threshold,  # Confidence threshold
            iou=iou_threshold,    # IoU threshold for NMS
            verbose=False
        )
        
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

