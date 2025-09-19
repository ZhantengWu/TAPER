"""
Our method: Step-by-step retry strategy for robust QR code reading.
Implements the core contribution of this project.
"""

import sys
import argparse
import cv2
import os
from pathlib import Path

os.environ['ZBAR_VERBOSITY'] = '0'

os.environ['ZBAR_DEBUG'] = '0'
os.environ['ZBAR_QUIET'] = '1'

sys.path.append(str(Path(__file__).parent.parent))

from utils.io import list_images, save_json, get_basename, load_yaml
from utils.timer import Timer
from decode.backend import get_default_backend
from decode.retry import retry_decode


def run_our_method(img_dir: str, cfg_path: str, output_json: str, time_budget_ms: float = None, verbose: bool = False) -> None:
    """
    Run our step-by-step retry method on a directory of images.
    
    Args:
        img_dir: Directory containing images
        cfg_path: Path to YAML configuration file
        output_json: Output JSON file path
        time_budget_ms: Time budget per image in milliseconds (overrides config)
        verbose: Whether to print progress
    """
    cfg = load_yaml(cfg_path)
    
    if time_budget_ms is not None:
        cfg['time_budget_ms'] = time_budget_ms
    
    print(f"Loaded configuration from: {cfg_path}")
    print(f"Time budget: {cfg.get('time_budget_ms', 120)} ms")
    
    image_files = list_images(img_dir)
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} images in {img_dir}")
    
    backend = get_default_backend()
    print(f"Using decoder backend: {backend.name}")
    
    enabled_steps = []
    if "steps" in cfg:
        for step in cfg["steps"]:
            if isinstance(step, dict):
                enabled_steps.append(step["name"])
            else:
                enabled_steps.append(step)

    if cfg.get("fallback_zxing", False):
        enabled_steps.append("zxing_fallback")
    
    print(f"Enabled steps: {' -> '.join(enabled_steps)}")
    
    results = []
    total_timer = Timer()
    total_timer.start()
    
    stage_counts = {}
    
    for i, img_path in enumerate(image_files):
        if verbose:
            print(f"Processing {i+1}/{len(image_files)}: {get_basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        texts, final_stage, elapsed_ms_val = retry_decode(img, backend, cfg)
        
        result = {
            "file": get_basename(img_path),
            "ok": len(texts) > 0,
            "texts": texts,
            "stage": final_stage,
            "ms": round(elapsed_ms_val, 1),
            "time_budget_ms": cfg.get("time_budget_ms", 120)  
        }
        results.append(result)
        
        stage_counts[final_stage] = stage_counts.get(final_stage, 0) + 1
        
        if verbose:
            if texts:
                print(f"  -> Success at stage '{final_stage}': {texts}")
            else:
                print(f"  -> Failed after {elapsed_ms_val:.1f} ms")
    
    total_time = total_timer.stop()
    
    successful = sum(1 for r in results if r["ok"])
    total_ms = sum(r["ms"] for r in results)
    avg_ms = total_ms / len(results) if results else 0
    
    results_with_meta = {
        "meta": {
            "method": "Our Method (Step-by-step Retry with Time Budget)",
            "time_budget_ms": cfg.get("time_budget_ms", 120),
            "enabled_steps": enabled_steps,
            "total_images": len(results),
            "successful_decodes": successful,
            "avg_time_ms": avg_ms,
            "stage_breakdown": stage_counts
        },
        "results": results
    }
    
    save_json(output_json, results_with_meta)
    fps = len(results) / total_time if total_time > 0 else 0
    
    print(f"\nOur Method Results:")
    print(f"  Total images: {len(results)}")
    print(f"  Successful decodes: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  Average time per image: {avg_ms:.1f} ms")
    print(f"  Processing speed: {fps:.1f} FPS")
    
    print(f"\nStage breakdown:")
    for stage, count in sorted(stage_counts.items()):
        percentage = count / len(results) * 100
        print(f"  {stage}: {count} ({percentage:.1f}%)")
    
    print(f"  Results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Our method: Step-by-step retry strategy")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    parser.add_argument("--cfg", required=True, help="YAML configuration file path")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--time_budget_ms", type=float, help="Time budget per image in milliseconds (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        run_our_method(args.img_dir, args.cfg, args.out, args.time_budget_ms, args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
