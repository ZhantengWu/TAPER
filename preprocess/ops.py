"""
Image preprocessing operations for QR code robust reading.
Includes upsampling, thresholding, deskewing, and rotation operations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Generator, Optional


def limit_image_size(img: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
    """
    Limit image size to stabilize processing time and ZXing hit rate.
    
    Args:
        img: Input image in BGR format
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Resized image if needed
    """
    height, width = img.shape[:2]
    
    if max(height, width) <= max_dimension:
        return img.copy()
    
    # Calculate scale factor
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def analyze_image_properties(img: np.ndarray) -> dict:
    """
    Analyze image properties for conditional processing decisions.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Dictionary with analysis results
    """

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    height, width = gray.shape
    
    contrast = gray.std()
    is_low_contrast = contrast < 40  
    
    min_qr_size = min(width, height) * 0.1 
    is_small_target = min_qr_size < 100  
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    estimated_skew = 0.0
    is_significantly_skewed = False
    is_barcode_like = False
    
    if lines is not None and len(lines) > 5:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            angles.append(angle)
        
        if angles:
            estimated_skew = np.median(angles)
            angle_std = np.std(angles)
            
            is_significantly_skewed = abs(estimated_skew) > 5.0 and angle_std < 10.0
            
            horizontal_lines = sum(1 for a in angles if abs(a) < 15)
            vertical_lines = sum(1 for a in angles if abs(abs(a) - 90) < 15)
            total_lines = len(angles)
            
            is_barcode_like = (horizontal_lines + vertical_lines) / total_lines > 0.7
    
    return {
        'contrast': contrast,
        'is_low_contrast': is_low_contrast,
        'is_small_target': is_small_target,
        'estimated_skew': estimated_skew,
        'is_significantly_skewed': is_significantly_skewed,
        'is_barcode_like': is_barcode_like,
        'width': width,
        'height': height
    }


def should_apply_threshold(properties: dict) -> bool:
    """Decide if threshold processing should be applied."""
    return properties['is_low_contrast']


def should_apply_deskew(properties: dict) -> bool:
    """Decide if deskew processing should be applied."""
    return (properties['is_significantly_skewed'] and 
            not properties['is_barcode_like'])


def should_apply_upsample(properties: dict) -> bool:
    """Decide if upsampling should be applied."""
    return properties['is_small_target']


def get_prioritized_rotation_angles(properties: dict, default_angles: List[float]) -> List[float]:
    """
    Get rotation angles prioritized by estimated skew angle.
    
    Args:
        properties: Image analysis properties
        default_angles: Default rotation angles from config
        
    Returns:
        Prioritized list of angles to try
    """
    estimated_skew = properties['estimated_skew']
    
    if abs(estimated_skew) < 2.0:
        return default_angles
    
    angles_with_distance = []
    for angle in default_angles:
        distance = abs(angle - estimated_skew)
        angles_with_distance.append((distance, angle))
    
    angles_with_distance.sort(key=lambda x: x[0])
    return [angle for _, angle in angles_with_distance]


def upsample(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Upsample image for better small target detection.
    
    Args:
        img: Input image in BGR format
        scale: Upsampling scale factor
        
    Returns:
        Upsampled image
    """
    if scale <= 1.0:
        return img.copy()
    
    height, width = img.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def auto_threshold(img: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding and brightness enhancement.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Processed image
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def deskew_light(img: np.ndarray, angle_threshold: float = 8.0) -> np.ndarray:
    """
    Conservative deskewing using Hough line detection.
    Strictly limited correction to avoid performance degradation.
    
    Args:
        img: Input image in BGR format
        angle_threshold: Maximum angle to correct (degrees) - much stricter now
        
    Returns:
        Deskewed image or original if correction not beneficial
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    edges = cv2.Canny(gray, 80, 200, apertureSize=3)
    
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    if lines is None or len(lines) < 5: 
        return img.copy()
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90  
        
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        
        if 2.0 <= abs(angle) <= angle_threshold:  
            angles.append(angle)
    
    if len(angles) < 3:  
        return img.copy()
    
    median_angle = np.median(angles)
    angle_std = np.std(angles)
    
    if angle_std > 3.0: 
        return img.copy()
    
    if abs(median_angle) < 2.0 or abs(median_angle) > angle_threshold:
        return img.copy()
    
    try:
        import pyzbar.pyzbar as pyzbar
        
        original_codes = pyzbar.decode(img)
        if len(original_codes) > 0:  
            return img.copy()
    except:
        pass  
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    corrected = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), 
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    
    return corrected


def rotate_try(img: np.ndarray, angles: List[float] = None) -> Generator[np.ndarray, None, None]:
    """
    Generate rotated versions of the image for multi-angle attempts.
    
    Args:
        img: Input image in BGR format
        angles: List of rotation angles in degrees
        
    Yields:
        Rotated image versions
    """
    if angles is None:
        angles = [-10, -5, 5, 10]
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    for angle in angles:
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        rotated = cv2.warpAffine(img, rotation_matrix, (new_width, new_height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
        
        yield rotated


def resize_for_speed(img: np.ndarray, max_dimension: int = 640) -> np.ndarray:
    """
    Resize image to speed up processing while maintaining aspect ratio.
    
    Args:
        img: Input image in BGR format
        max_dimension: Maximum dimension (width or height)
        
    Returns:
        Resized image
    """
    height, width = img.shape[:2]
    
    if max(height, width) <= max_dimension:
        return img.copy()
    
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast for better QR code detection.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Contrast-enhanced image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply sharpening filter to improve edge definition.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    
    return cv2.filter2D(img, -1, kernel)
