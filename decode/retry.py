"""
Core retry strategy for robust QR code reading.
Implements step-by-step retry with early stopping and time budget control.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.timer import now_ms, elapsed_ms
from decode.backend import DecoderBackend, create_backend
from preprocess.ops import upsample, auto_threshold, deskew_light, rotate_try


def has_step(cfg: Dict[str, Any], step_name: str) -> bool:
    """
    Check if a step is enabled in configuration.
    
    Args:
        cfg: Configuration dictionary
        step_name: Name of the step to check
        
    Returns:
        True if step is enabled
    """
    if "steps" not in cfg:
        return False
    
    for step in cfg["steps"]:
        if isinstance(step, dict) and step.get("name") == step_name:
            return True
        elif isinstance(step, str) and step == step_name:
            return True
    
    return False


def get_step_config(cfg: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific step.
    
    Args:
        cfg: Configuration dictionary
        step_name: Name of the step
        
    Returns:
        Step configuration dictionary
    """
    if "steps" not in cfg:
        return {}
    
    for step in cfg["steps"]:
        if isinstance(step, dict) and step.get("name") == step_name:
            return step
        elif isinstance(step, str) and step == step_name:
            return {"name": step_name}
    
    return {}


def safe_decode(backend: DecoderBackend, img: np.ndarray) -> List[str]:
    """
    Safe wrapper for backend decoding that handles zbar assertion errors.
    
    Args:
        backend: Decoder backend to use
        img: Input image
        
    Returns:
        List of decoded texts, empty if failed
    """
    try:
        return backend.decode(img)
    except Exception as e:
        error_msg = str(e).lower()
        if "zbar" in error_msg and "assertion" in error_msg:
            # Known zbar assertion error - try with different image preprocessing
            try:
                # Try with different image format/preprocessing
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img
                
                # Apply gentle smoothing to avoid assertion triggers
                smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
                
                # Try decoding the smoothed version
                return backend.decode(smoothed)
            except:
                # If still fails, return empty
                return []
        else:
            # Other errors, re-raise
            raise e


def retry_decode(img_bgr: np.ndarray, backend: DecoderBackend, cfg: Dict[str, Any]) -> Tuple[List[str], str, float]:
    """
    Apply intelligent conditional retry strategy for robust QR code decoding.
    
    This implements conditional triggering pipeline:
    - Automatic image analysis to determine which steps are needed
    - Smart rotation angle prioritization based on estimated skew
    - Early stopping when successful
    - Image size limiting for stable performance
    
    Args:
        img_bgr: Input image in BGR format
        backend: Decoder backend to use
        cfg: Configuration dictionary from YAML
        
    Returns:
        Tuple of (decoded_texts, final_stage, total_time_ms)
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from preprocess.ops import (
        limit_image_size, analyze_image_properties, 
        should_apply_threshold, should_apply_deskew, should_apply_upsample,
        get_prioritized_rotation_angles
    )
    
    t0 = now_ms()
    time_budget_ms = cfg.get("time_budget_ms", 120)
    
    # Limit image size for stable performance
    current_img = limit_image_size(img_bgr.copy(), max_dimension=2000)
    
    # Analyze image properties for conditional processing
    properties = analyze_image_properties(current_img)
    
    # Step 1: Direct decoding
    if elapsed_ms(t0) > time_budget_ms:
        return [], "timeout", elapsed_ms(t0)
    
    texts = safe_decode(backend, current_img)
    if texts:
        return texts, "direct", elapsed_ms(t0)
    
    # Step 2: Conditional Threshold (only for low contrast)
    if should_apply_threshold(properties):
        if elapsed_ms(t0) > time_budget_ms:
            return [], "timeout", elapsed_ms(t0)
        
        from preprocess.ops import auto_threshold
        threshold_img = auto_threshold(current_img)
        texts = safe_decode(backend, threshold_img)
        if texts:
            return texts, "threshold", elapsed_ms(t0)
    
    # Step 3: Smart Rotation (prioritized by estimated skew angle)
    steps = cfg.get("steps", [])
    rot_try_step = None
    for step in steps:
        if step.get("name") == "rot_try":
            rot_try_step = step
            break
    
    if rot_try_step:
        default_angles = rot_try_step.get("angles", [-10, -5, 5, 10])
        prioritized_angles = get_prioritized_rotation_angles(properties, default_angles)
        
        from preprocess.ops import rotate_try
        for rotated_img in rotate_try(current_img, prioritized_angles):
            if elapsed_ms(t0) > time_budget_ms:
                return [], "timeout", elapsed_ms(t0)
            
            texts = safe_decode(backend, rotated_img)
            if texts:
                return texts, "rot_try", elapsed_ms(t0)
    
    # Step 4: Conditional Deskew (only for significantly skewed non-barcode images)
    if should_apply_deskew(properties):
        if elapsed_ms(t0) > time_budget_ms:
            return [], "timeout", elapsed_ms(t0)
        
        from preprocess.ops import deskew_light
        deskewed_img = deskew_light(current_img)
        texts = safe_decode(backend, deskewed_img)
        if texts:
            return texts, "deskew", elapsed_ms(t0)
    
    # Step 5: Conditional Upsample (only for small targets)
    if should_apply_upsample(properties):
        if elapsed_ms(t0) > time_budget_ms:
            return [], "timeout", elapsed_ms(t0)
        
        from preprocess.ops import upsample
        upsampled_img = upsample(current_img, scale=1.5)
        texts = safe_decode(backend, upsampled_img)
        if texts:
            return texts, "upsample", elapsed_ms(t0)
    
    # Step 6: Fallback to ZXing if enabled and time permits
    if cfg.get("fallback_zxing", False) and elapsed_ms(t0) <= time_budget_ms:
        try:
            import pyzbar.pyzbar as pyzbar
            codes = pyzbar.decode(current_img)
            if codes:
                texts = [code.data.decode('utf-8') for code in codes]
                return texts, "zxing_fallback", elapsed_ms(t0)
        except:
            pass
    
    # All steps failed
    return [], "fail", elapsed_ms(t0)


def retry_decode_with_stages(img_bgr: np.ndarray, backend: DecoderBackend, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply retry strategy and return detailed stage information.

    Args:
        img_bgr: Input image in BGR format
        backend: Decoder backend to use
        cfg: Configuration dictionary from YAML

    Returns:
        Dictionary with detailed results including stage timings
    """
    t0 = now_ms()
    time_budget_ms = cfg.get("time_budget_ms", 120)

    stage_results = []
    current_img = img_bgr.copy()

    # Step 1: Direct decoding
    stage_start = now_ms()
    texts = backend.decode(current_img)
    stage_time = elapsed_ms(stage_start)

    stage_results.append({
        "stage": "direct",
        "success": len(texts) > 0,
        "texts": texts,
        "time_ms": stage_time
    })

    if texts:
        return {
            "texts": texts,
            "final_stage": "direct",
            "total_time_ms": elapsed_ms(t0),
            "stages": stage_results
        }

    # Step 2: Upsample
    if has_step(cfg, "upsample"):
        stage_start = now_ms()
        step_cfg = get_step_config(cfg, "upsample")
        scale = step_cfg.get("scale", 1.5)

        current_img = upsample(img_bgr, scale)
        texts = backend.decode(current_img)
        stage_time = elapsed_ms(stage_start)

        stage_results.append({
            "stage": "upsample",
            "success": len(texts) > 0,
            "texts": texts,
            "time_ms": stage_time,
            "scale": scale
        })

        if texts:
            return {
                "texts": texts,
                "final_stage": "upsample",
                "total_time_ms": elapsed_ms(t0),
                "stages": stage_results
            }

    # Step 3: Threshold
    if has_step(cfg, "threshold"):
        stage_start = now_ms()
        img_thresh = auto_threshold(current_img)
        texts = backend.decode(img_thresh)
        stage_time = elapsed_ms(stage_start)

        stage_results.append({
            "stage": "threshold",
            "success": len(texts) > 0,
            "texts": texts,
            "time_ms": stage_time
        })

        if texts:
            return {
                "texts": texts,
                "final_stage": "threshold",
                "total_time_ms": elapsed_ms(t0),
                "stages": stage_results
            }

        current_img = img_thresh

    # Step 4: Deskew
    if has_step(cfg, "deskew"):
        stage_start = now_ms()
        img_deskewed = deskew_light(current_img)
        texts = backend.decode(img_deskewed)
        stage_time = elapsed_ms(stage_start)

        stage_results.append({
            "stage": "deskew",
            "success": len(texts) > 0,
            "texts": texts,
            "time_ms": stage_time
        })

        if texts:
            return {
                "texts": texts,
                "final_stage": "deskew",
                "total_time_ms": elapsed_ms(t0),
                "stages": stage_results
            }

        current_img = img_deskewed

    # Step 5: Rotation attempts
    if has_step(cfg, "rot_try"):
        stage_start = now_ms()
        step_cfg = get_step_config(cfg, "rot_try")
        angles = step_cfg.get("angles", [-10, -5, 5, 10])

        rotation_attempts = []
        for angle in angles:
            if elapsed_ms(t0) > time_budget_ms:
                break

            attempt_start = now_ms()
            for rotated_img in rotate_try(current_img, [angle]):
                texts = backend.decode(rotated_img)
                attempt_time = elapsed_ms(attempt_start)

                rotation_attempts.append({
                    "angle": angle,
                    "success": len(texts) > 0,
                    "texts": texts,
                    "time_ms": attempt_time
                })

                if texts:
                    stage_results.append({
                        "stage": "rot_try",
                        "success": True,
                        "texts": texts,
                        "time_ms": elapsed_ms(stage_start),
                        "successful_angle": angle,
                        "attempts": rotation_attempts
                    })

                    return {
                        "texts": texts,
                        "final_stage": "rot_try",
                        "total_time_ms": elapsed_ms(t0),
                        "stages": stage_results
                    }
                break

        stage_results.append({
            "stage": "rot_try",
            "success": False,
            "texts": [],
            "time_ms": elapsed_ms(stage_start),
            "attempts": rotation_attempts
        })

    # Step 6: ZXing fallback
    if cfg.get("fallback_zxing", False):
        stage_start = now_ms()
        try:
            zxing_backend = create_backend("zxing")
            texts = zxing_backend.decode(current_img)
            stage_time = elapsed_ms(stage_start)

            stage_results.append({
                "stage": "zxing",
                "success": len(texts) > 0,
                "texts": texts,
                "time_ms": stage_time
            })

            if texts:
                return {
                    "texts": texts,
                    "final_stage": "zxing",
                    "total_time_ms": elapsed_ms(t0),
                    "stages": stage_results
                }
        except Exception as e:
                stage_results.append({
                    "stage": "zxing",
                    "success": False,
                    "texts": [],
                    "time_ms": elapsed_ms(stage_start),
                    "error": str(e)
                })

    # All stages failed
    return {
        "texts": [],
        "final_stage": "fail",
        "total_time_ms": elapsed_ms(t0),
        "stages": stage_results
    }
