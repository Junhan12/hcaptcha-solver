"""
Modular preprocessing functions for image processing operations.
Each operation is implemented as a separate function that can be chained together.
"""
import numpy as np
from io import BytesIO
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _bytes_to_cv2_image(img_bytes):
    """
    Convert image bytes to OpenCV format (numpy array).
    Uses IMREAD_UNCHANGED to preserve original format (including alpha channel).
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for preprocessing")
    nparr = np.frombuffer(img_bytes, np.uint8)
    # Use IMREAD_UNCHANGED to preserve original format (RGBA, BGRA, grayscale, etc.)
    # This matches the notebook behavior which uses IMREAD_UNCHANGED
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _cv2_image_to_bytes(img_cv2, format='JPG'):
    """Convert OpenCV image (numpy array) back to bytes."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for preprocessing")
    success, encoded_img = cv2.imencode(f'.{format.lower()}', img_cv2)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_img.tobytes()


# =========================
# Individual preprocessing operations
# =========================

def apply_bilateral(img, params):
    """
    Apply bilateral filter for edge-preserving smoothing.
    params: {"d": int, "sigmaColor": float, "sigmaSpace": float}
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for bilateral filter")
    d = params.get("d", 9)
    sigmaColor = params.get("sigmaColor", 75)
    sigmaSpace = params.get("sigmaSpace", 75)
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def apply_median(img, params):
    """
    Apply median blur for noise reduction.
    params: {"ksize": int} (must be odd, >= 3)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for median blur")
    ksize = params.get("ksize", 5)
    # Ensure ksize is odd
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    return cv2.medianBlur(img, ksize)


def apply_gaussian(img, params):
    """
    Apply Gaussian blur.
    params: {"ksize": [int, int] or int, "sigmaX": float}
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for Gaussian blur")
    ksize = params.get("ksize", [3, 3])
    if isinstance(ksize, list):
        ksize = tuple(ksize)
    elif isinstance(ksize, int):
        # Ensure odd
        if ksize % 2 == 0:
            ksize += 1
        ksize = (ksize, ksize)
    sigmaX = params.get("sigmaX", 0)
    return cv2.GaussianBlur(img, ksize, sigmaX)


def apply_nlmeans(img, params):
    """
    Apply Non-Local Means denoising.
    Supports both color and grayscale images.
    params: {"h": float, "templateWindowSize": int, "searchWindowSize": int}
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for NL-Means denoising")
    
    h = params.get("h", 10)
    hColor = params.get("hColor", 5)
    templateWindowSize = params.get("templateWindowSize", 7)
    searchWindowSize = params.get("searchWindowSize", 21)
    
    # Check image shape to determine if it's grayscale or color
    if len(img.shape) == 2:
        # Grayscale image (2D array) - use fastNlMeansDenoising
        return cv2.fastNlMeansDenoising(
            img, 
            None, 
            h, 
            templateWindowSize, 
            searchWindowSize
        )
    elif len(img.shape) == 3:
        # Color image (3D array)
        num_channels = img.shape[2]
        if num_channels == 3:
            # BGR color image (3 channels) - use fastNlMeansDenoisingColored
            return cv2.fastNlMeansDenoisingColored(
                img, 
                None, 
                h, 
                hColor, 
                templateWindowSize, 
                searchWindowSize
            )
        elif num_channels == 4:
            # BGRA color image (4 channels) - use fastNlMeansDenoisingColored
            return cv2.fastNlMeansDenoisingColored(
                img, 
                None, 
                h, 
                hColor, 
                templateWindowSize, 
                searchWindowSize
            )
        else:
            # Unexpected number of channels, convert to grayscale and denoise
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if num_channels > 1 else img
            return cv2.fastNlMeansDenoising(
                gray, 
                None, 
                h, 
                templateWindowSize, 
                searchWindowSize
            )
    else:
        # Unexpected image format, return as-is
        print(f"Warning: Unexpected image shape {img.shape} for NL-Means denoising")
        return img


def apply_laplacian(img, params):
    """
    Apply Laplacian edge detection.
    params: {"ddepth": int} (typically cv2.CV_64F or 64)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for Laplacian")
    ddepth = params.get("ddepth", cv2.CV_64F)
    laplacian = cv2.Laplacian(img, ddepth)
    # Convert back to uint8 for display/processing
    laplacian_abs = np.absolute(laplacian)
    return np.uint8(laplacian_abs)


def apply_addweighted(img, params):
    """
    Blend two images using weighted addition.
    Since we only have one image, we blend it with itself using different weights.
    params: {"alpha": float, "beta": float, "gamma": float}
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for addWeighted")
    alpha = params.get("alpha", 0.7)
    beta = params.get("beta", 0.3)
    gamma = params.get("gamma", 0)
    # Blend original with itself (can be modified to blend with a processed version)
    return cv2.addWeighted(img, alpha, img, beta, gamma)

def apply_sharpen(img, params):
    """
    Apply sharpening filter using a 3x3 kernel.
    params: {} (no parameters needed, uses fixed kernel)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for sharpening")
    
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

    return cv2.filter2D(img, -1, kernel_sharpen)


def apply_grayscale(img, params):
    """
    Convert image to grayscale.
    Matches notebook behavior: uses COLOR_BGRA2GRAY for 4-channel images.
    params: {} (no parameters needed, but can accept empty dict)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for grayscale conversion")
    
    # Check if image is already grayscale (2D array)
    if len(img.shape) == 2:
        return img
    
    # Check image channels and convert appropriately
    if len(img.shape) == 3:
        num_channels = img.shape[2]
        
        # BGRA (4 channels) - matches notebook behavior
        if num_channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # BGR (3 channels)
        elif num_channels == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # RGBA (4 channels) - handle if needed
        # Note: OpenCV uses BGR by default, but if image is RGBA, 
        # we'd need to convert BGR first, but IMREAD_UNCHANGED should preserve format
    
    # If already grayscale or unknown format, return as is
    return img


def resize_labels(label_content, resize_info):
    """
    Resize YOLO format labels based on image resize information.
    
    Args:
        label_content: str - Content of YOLO label file (each line: class_id x_center y_center width height)
        resize_info: dict - Resize information from apply_resize function
    
    Returns:
        str - Resized label content
    """
    if not resize_info or resize_info.get("scale_w") == 1.0 and resize_info.get("scale_h") == 1.0:
        # No resize occurred, return original
        return label_content
    
    scale_w = resize_info.get("scale_w", 1.0)
    scale_h = resize_info.get("scale_h", 1.0)
    maintain_aspect = resize_info.get("maintain_aspect", False)
    
    # If maintain_aspect is True, use uniform scale
    if maintain_aspect:
        scale = min(scale_w, scale_h)
        scale_w = scale
        scale_h = scale
    
    lines = label_content.strip().split('\n')
    resized_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            resized_lines.append(line)
            continue
        
        parts = line.split()
        if len(parts) < 5:
            # Invalid line, keep as is
            resized_lines.append(line)
            continue
        
        try:
            class_id = parts[0]
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Apply scaling to normalized coordinates
            x_center_new = x_center * scale_w
            y_center_new = y_center * scale_h
            width_new = width * scale_w
            height_new = height * scale_h
            
            # Ensure coordinates stay within [0, 1] bounds
            x_center_new = max(0.0, min(1.0, x_center_new))
            y_center_new = max(0.0, min(1.0, y_center_new))
            width_new = max(0.0, min(1.0, width_new))
            height_new = max(0.0, min(1.0, height_new))
            
            # Reconstruct line
            resized_line = f"{class_id} {x_center_new:.6f} {y_center_new:.6f} {width_new:.6f} {height_new:.6f}"
            resized_lines.append(resized_line)
            
        except (ValueError, IndexError):
            # Invalid format, keep original line
            resized_lines.append(line)
    
    return '\n'.join(resized_lines)


def apply_resize(img, params):
    """
    Resize image to specified dimensions.
    params: {
        "width": int (target width in pixels),
        "height": int (target height in pixels),
        "maintain_aspect": bool (default: False, if True, calculates 
                                  missing dimension to preserve aspect ratio),
        "interpolation": str (default: "linear", options: "linear", "cubic", 
                              "area", "nearest", "lanczos")
    }
    
    Examples:
        - {"width": 640, "height": 480} - resize to exact dimensions
        - {"width": 640, "maintain_aspect": true} - resize width, auto height
        - {"height": 480, "maintain_aspect": true} - resize height, auto width
    
    Returns:
        tuple: (resized_img, resize_info)
        - resized_img: The resized image
        - resize_info: dict with keys:
            - "original_width": int
            - "original_height": int
            - "new_width": int
            - "new_height": int
            - "scale_w": float (width scale factor)
            - "scale_h": float (height scale factor)
            - "maintain_aspect": bool
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for resize")
    
    # Validate image shape
    if img is None:
        raise ValueError("Image is None, cannot resize")
    if not hasattr(img, 'shape') or len(img.shape) < 2:
        raise ValueError(f"Invalid image shape: {img.shape if hasattr(img, 'shape') else 'no shape attribute'}")
    
    # Get current image dimensions
    # Handle both 2D (grayscale) and 3D (color) images
    if len(img.shape) == 2:
        # Grayscale image: shape is (height, width)
        h, w = img.shape
    elif len(img.shape) == 3:
        # Color image: shape is (height, width, channels)
        h, w = img.shape[:2]
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}, expected 2D or 3D array")
    
    original_w, original_h = w, h
    
    # Get target dimensions
    target_width = params.get("width")
    target_height = params.get("height")
    maintain_aspect = params.get("maintain_aspect", False)
    
    # Determine interpolation method
    interpolation_map = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation_str = params.get("interpolation", "linear").lower()
    interpolation = interpolation_map.get(interpolation_str, cv2.INTER_LINEAR)
    
    # Calculate target dimensions
    if target_width is None and target_height is None:
        # No dimensions specified, return original
        resize_info = {
            "original_width": original_w,
            "original_height": original_h,
            "new_width": original_w,
            "new_height": original_h,
            "scale_w": 1.0,
            "scale_h": 1.0,
            "maintain_aspect": maintain_aspect
        }
        return img, resize_info
    
    if maintain_aspect:
        # Calculate missing dimension to maintain aspect ratio
        if target_width is None and target_height is not None:
            # Calculate width from height
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
        elif target_height is None and target_width is not None:
            # Calculate height from width
            aspect_ratio = h / w
            target_height = int(target_width * aspect_ratio)
        elif target_width is not None and target_height is not None:
            # Both specified, but maintain aspect - use the dimension that
            # requires less scaling
            scale_w = target_width / w
            scale_h = target_height / h
            scale = min(scale_w, scale_h)
            target_width = int(w * scale)
            target_height = int(h * scale)
    else:
        # Use exact dimensions (may distort aspect ratio)
        if target_width is None:
            target_width = w
        if target_height is None:
            target_height = h
    
    # Ensure dimensions are positive
    target_width = max(1, target_width)
    target_height = max(1, target_height)
    
    # Calculate scale factors
    scale_w = target_width / original_w
    scale_h = target_height / original_h
    
    # If maintain_aspect is True, use uniform scale
    if maintain_aspect:
        scale = min(scale_w, scale_h)
        scale_w = scale
        scale_h = scale
    
    # Perform resize
    resized_img = cv2.resize(img, (target_width, target_height), 
                              interpolation=interpolation)
    
    # Create resize info
    resize_info = {
        "original_width": original_w,
        "original_height": original_h,
        "new_width": target_width,
        "new_height": target_height,
        "scale_w": scale_w,
        "scale_h": scale_h,
        "maintain_aspect": maintain_aspect
    }
    
    return resized_img, resize_info


# Operation registry
OPERATION_REGISTRY = {
    "bilateral": apply_bilateral,
    "median": apply_median,
    "gaussian": apply_gaussian,
    "nlmeans": apply_nlmeans,
    "laplacian": apply_laplacian,
    "addweighted": apply_addweighted,
    "grayscale": apply_grayscale,
    "sharpen": apply_sharpen,          
    "resize": apply_resize,
}


# =========================
# Main preprocessing pipeline
# =========================

def apply_preprocess(img_bytes, preprocess_profile):
    """
    Apply preprocessing steps from a preprocess_profile to image bytes.
    
    Args:
        img_bytes: Raw image bytes (JPEG/PNG)
        preprocess_profile: Dict with 'steps' array, each step has 'operation' and 'params'
    
    Returns:
        tuple: (processed_img_bytes, applied_steps_info)
        - processed_img_bytes: Processed image as bytes
        - applied_steps_info: List of dicts with operation names that were applied
    """
    if not CV2_AVAILABLE:
        return img_bytes, [], None
    
    if not preprocess_profile:
        return img_bytes, [], None
    
    steps = preprocess_profile.get("steps", [])
    if not steps:
        return img_bytes, [], None
    
    try:
        # Convert bytes to OpenCV image
        img = _bytes_to_cv2_image(img_bytes)
        applied = []
        
        # Track resize info for label processing
        resize_info = None
        
        # Apply each step in sequence
        for step in steps:
            operation_name = step.get("operation")
            params = step.get("params", {})
            
            if operation_name not in OPERATION_REGISTRY:
                print(f"Warning: Unknown operation '{operation_name}', skipping")
                continue
            
            try:
                operation_func = OPERATION_REGISTRY[operation_name]
                
                # Special handling for resize operation (returns tuple)
                if operation_name == "resize":
                    img, resize_info = operation_func(img, params)
                else:
                img = operation_func(img, params)
                
                applied.append({"operation": operation_name, "params": params})
            except Exception as e:
                print(f"Error applying {operation_name}: {e}")
                # Continue with next step on error
                continue
        
        # Convert back to bytes
        processed_bytes = _cv2_image_to_bytes(img, format='JPG')
        return processed_bytes, applied, resize_info
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Return original bytes on failure
        return img_bytes, [], None


