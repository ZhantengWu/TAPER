"""
Baseline E: Direct QReader decoding without any preprocessing.
QReader is a modern deep learning-based QR code detector and decoder.
"""

import sys
import argparse
import cv2
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import list_images, save_json, get_basename
from utils.timer import Timer
from decode.backend import create_backend


def run_baseline_e(img_dir: str, output_json: str, time_budget_ms: float = None, verbose: bool = False) -> None:
    """
    Run baseline E (direct QReader decoding) on a directory of images.
    
    Args:
        img_dir: Directory containing images
        output_json: Output JSON file path
        verbose: Whether to print progress
    """
    # Get image files
    image_files = list_images(img_dir)
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} images in {img_dir}")
    
    # Initialize QReader backend
    try:
        backend = create_backend("qreader")
        print(f"Using decoder backend: {backend.name}")
    except Exception as e:
        print(f"Error initializing QReader backend: {e}")
        print("Make sure qreader is installed: pip install qreader")
        return
    
    # Process images
    results = []
    total_timer = Timer()
    total_timer.start()
    
    for i, img_path in enumerate(image_files):
        if verbose:
            print(f"Processing {i+1}/{len(image_files)}: {get_basename(img_path)}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            continue
        
        # Time the decoding
        timer = Timer()
        timer.start()
        
        # Direct QReader decoding (baseline E)
        texts = backend.decode(img)
        
        elapsed_ms = timer.stop() * 1000.0
        
        # Check time budget if specified
        stage = "direct_qreader"
        is_timeout = time_budget_ms and elapsed_ms > time_budget_ms
        if is_timeout:
            stage = "timeout"
            if verbose:
                print(f"  -> Timeout: {elapsed_ms:.1f}ms > {time_budget_ms}ms")
        
        # Record result - 超时时不算成功
        result = {
            "file": get_basename(img_path),
            "ok": len(texts) > 0 and not is_timeout,
            "texts": texts if not is_timeout else [],
            "stage": stage,
            "ms": round(elapsed_ms, 1)
        }
        results.append(result)
        
        if verbose and texts:
            print(f"  -> Decoded: {texts}")
    
    total_time = total_timer.stop()
    
    # Save results
    save_json(output_json, results)
    
    # Print summary
    successful = sum(1 for r in results if r["ok"])
    total_ms = sum(r["ms"] for r in results)
    avg_ms = total_ms / len(results) if results else 0
    fps = len(results) / total_time if total_time > 0 else 0
    
    print(f"\nBaseline E (QReader) Results:")
    print(f"  Total images: {len(results)}")
    print(f"  Successful decodes: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  Average time per image: {avg_ms:.1f} ms")
    print(f"  Processing speed: {fps:.1f} FPS")
    print(f"  Results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Baseline E: Direct QReader decoding")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--time_budget_ms", type=float, help="Time budget in milliseconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        run_baseline_e(args.img_dir, args.out, args.time_budget_ms, args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
