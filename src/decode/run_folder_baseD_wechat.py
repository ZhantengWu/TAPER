"""
Baseline D: WeChatQRCode decoding without any preprocessing.
This provides WeChatQRCode baseline using OpenCV for comparison.
"""

import sys
import argparse
import cv2
from pathlib import Path

# Try to import wechat_qrcode module (for new API compatibility)
try:
    import cv2.wechat_qrcode
except ImportError:
    pass  # Will fallback to old API

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import list_images, save_json, get_basename
from utils.timer import Timer


def run_baseline_d(img_dir: str, output_json: str, model_dir: str = "models/wechatqr", time_budget_ms: float = None, verbose: bool = False) -> None:
    """
    Run baseline D (WeChatQRCode decoding) on a directory of images.
    
    Args:
        img_dir: Directory containing images
        output_json: Output JSON file path
        model_dir: WeChatQRCode model files directory
        time_budget_ms: Time budget per image in milliseconds (optional)
        verbose: Whether to print progress
    """
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    # Resolve model directory path
    if not Path(model_dir).is_absolute():
        model_dir = project_root / model_dir
    else:
        model_dir = Path(model_dir)
    
    # Model file paths
    det_p = str(model_dir / "detect.prototxt")
    det_m = str(model_dir / "detect.caffemodel")
    sr_p = str(model_dir / "sr.prototxt")
    sr_m = str(model_dir / "sr.caffemodel")
    
    # Check model files exist
    model_files = [det_p, det_m, sr_p, sr_m]
    for model_file in model_files:
        if not Path(model_file).exists():
            print(f"Error: Model file not found: {model_file}")
            print("Please download WeChatQRCode model files to models/wechatqr/ directory")
            return
    
    # Initialize WeChatQRCode detector with API compatibility
    wechat_qr = None
    api_version = "unknown"
    
    try:
        # Check OpenCV version
        cv_version = getattr(cv2, '__version__', 'unknown')
        
        # Try new API first (OpenCV 4.5.2+)
        if hasattr(cv2, 'wechat_qrcode') and hasattr(cv2.wechat_qrcode, 'WeChatQRCode'):
            try:
                wechat_qr = cv2.wechat_qrcode.WeChatQRCode(det_p, det_m, sr_p, sr_m)
                api_version = "new (cv2.wechat_qrcode.WeChatQRCode)"
                print(f"✅ Using new WeChatQRCode API (OpenCV {cv_version})")
            except Exception as e:
                print(f"⚠️  New API failed: {e}")
                wechat_qr = None
        
        # Fallback to old API (OpenCV 4.5.1 and earlier)
        if wechat_qr is None and hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
            try:
                wechat_qr = cv2.wechat_qrcode_WeChatQRCode(det_p, det_m, sr_p, sr_m)
                api_version = "old (cv2.wechat_qrcode_WeChatQRCode)"
                print(f"✅ Using legacy WeChatQRCode API (OpenCV {cv_version})")
            except Exception as e:
                print(f"⚠️  Legacy API failed: {e}")
                wechat_qr = None
        
        # Check if any API worked
        if wechat_qr is None:
            print(f"❌ WeChatQRCode not available in OpenCV {cv_version}")
            print("Please install opencv-contrib-python with WeChatQRCode support")
            print("Supported APIs:")
            print("  - New: cv2.wechat_qrcode.WeChatQRCode (OpenCV 4.5.2+)")
            print("  - Old: cv2.wechat_qrcode_WeChatQRCode (OpenCV 4.5.1-)")
            return
        
        print(f"Using decoder backend: WeChatQRCode {api_version}")
        
    except Exception as e:
        print(f"❌ Error initializing WeChatQRCode backend: {e}")
        return
    
    # Get image files
    image_files = list_images(img_dir)
    if not image_files:
        print(f"No images found in {img_dir}")
        return
    
    print(f"Found {len(image_files)} images in {img_dir}")

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
        
        # WeChatQRCode decoding (baseline D)
        try:
            texts, _ = wechat_qr.detectAndDecode(img)
            # Filter out empty strings
            texts = [t for t in texts if t] if texts else []
        except Exception as e:
            print(f"Warning: WeChatQRCode decoding failed for {get_basename(img_path)}: {e}")
            texts = []
        
        elapsed_ms = timer.stop() * 1000.0
        
        # Check time budget if specified
        stage = "direct_wechat"
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
    
    print(f"\nBaseline D (WeChatQRCode) Results:")
    print(f"  Total images: {len(results)}")
    print(f"  Successful decodes: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  Average time per image: {avg_ms:.1f} ms")
    print(f"  Processing speed: {fps:.1f} FPS")
    print(f"  Results saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Baseline D: WeChatQRCode decoding")
    parser.add_argument("--img_dir", required=True, help="Directory containing images")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--model_dir", default="models/wechatqr", help="WeChatQRCode model files directory")
    parser.add_argument("--time_budget_ms", type=float, help="Time budget per image in milliseconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        run_baseline_d(args.img_dir, args.out, args.model_dir, args.time_budget_ms, args.verbose)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
