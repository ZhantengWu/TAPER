"""
I/O utilities for QR code robust reading project.
Handles image file scanning, YAML config loading, JSON result saving.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


def list_images(img_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')) -> List[str]:
    """
    Scan directory for image files.
    
    Args:
        img_dir: Directory path containing images
        extensions: Supported image file extensions
        
    Returns:
        List of full image file paths
    """
    img_dir = Path(img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(img_dir.glob(f"*{ext}"))
        image_files.extend(img_dir.glob(f"*{ext.upper()}"))
    
    unique_files = list(dict.fromkeys(image_files))
    
    return [str(f) for f in sorted(unique_files)]


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary containing configuration
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(json_path: str, data: List[Dict[str, Any]], indent: int = 2) -> None:
    """
    Save results to JSON file.
    
    Args:
        json_path: Output JSON file path
        data: List of result dictionaries
        indent: JSON formatting indent
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load results from JSON file.
    
    Args:
        json_path: Input JSON file path
        
    Returns:
        List of result dictionaries
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    if isinstance(raw_data, dict) and "results" in raw_data:
        return raw_data["results"]
    else:
        return raw_data


def load_ground_truth(gt_dir: str, filename: str) -> Optional[str]:
    """
    Load ground truth text for a given image file.
    
    Args:
        gt_dir: Directory containing ground truth .txt files
        filename: Image filename (without extension)
        
    Returns:
        Ground truth text or None if not found
    """
    if not gt_dir:
        return None
    
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        return None
    
    base_name = Path(filename).stem
    gt_candidates = [
        gt_dir / f"{base_name}.txt",
        gt_dir / f"{filename}.txt",
    ]
    
    for gt_file in gt_candidates:
        if gt_file.exists():
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                continue
    
    return None


def ensure_dir(dir_path: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        dir_path: Directory path to create
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_basename(file_path: str) -> str:
    """
    Get basename of file path.
    
    Args:
        file_path: Full file path
        
    Returns:
        Basename without directory
    """
    return Path(file_path).name
