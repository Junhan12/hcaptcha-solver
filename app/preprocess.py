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
    """Convert image bytes to OpenCV format (numpy array)."""
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for preprocessing")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _cv2_image_to_bytes(img_cv2, format='PNG'):
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
    params: {} (no parameters needed, but can accept empty dict)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for grayscale conversion")
    
    # Check if image is already grayscale (2D array)
    if len(img.shape) == 2:
        return img
    
    # Check if image is BGR (3 channels)
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if image is BGRA (4 channels)
    if len(img.shape) == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    # If already grayscale or unknown format, return as is
    return img


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
        return img_bytes, []
    
    if not preprocess_profile:
        return img_bytes, []
    
    steps = preprocess_profile.get("steps", [])
    if not steps:
        return img_bytes, []
    
    try:
        # Convert bytes to OpenCV image
        img = _bytes_to_cv2_image(img_bytes)
        applied = []
        
        # Apply each step in sequence
        for step in steps:
            operation_name = step.get("operation")
            params = step.get("params", {})
            
            if operation_name not in OPERATION_REGISTRY:
                print(f"Warning: Unknown operation '{operation_name}', skipping")
                continue
            
            try:
                operation_func = OPERATION_REGISTRY[operation_name]
                img = operation_func(img, params)
                applied.append({"operation": operation_name, "params": params})
            except Exception as e:
                print(f"Error applying {operation_name}: {e}")
                # Continue with next step on error
                continue
        
        # Convert back to bytes
        processed_bytes = _cv2_image_to_bytes(img, format='PNG')
        return processed_bytes, applied
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        # Return original bytes on failure
        return img_bytes, []


