"""
Evaluation metrics for QR code robust reading project.
Calculates decode success rate, misread rate, and processing speed metrics.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import load_json, load_ground_truth


def calculate_metrics(results: List[Dict[str, Any]], gt_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from results.
    
    Args:
        results: List of result dictionaries from JSON
        gt_dir: Directory containing ground truth files (optional)
        
    Returns:
        Dictionary containing calculated metrics
    """
    if not results:
        return {
            "total_images": 0,
            "decode_success_rate": 0.0,
            "misread_rate": None,
            "avg_time_ms": 0.0,
            "fps": 0.0,
            "stage_breakdown": {}
        }
    
    total_images = len(results)
    successful_decodes = sum(1 for r in results if r.get("ok", False))
    
    # Calculate decode success rate
    decode_success_rate = successful_decodes / total_images * 100.0
    
    # Calculate timing metrics
    total_time_ms = sum(r.get("ms", 0) for r in results)
    avg_time_ms = total_time_ms / total_images
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
    
    # Calculate stage breakdown
    stage_counts = {}
    for result in results:
        stage = result.get("stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    # Calculate misread rate if ground truth is available
    misread_rate = None
    misread_details = None
    
    if gt_dir:
        misread_count = 0
        total_with_gt = 0
        misread_cases = []
        
        for result in results:
            filename = result.get("file", "")
            if not filename:
                continue
            
            # Load ground truth
            gt_text = load_ground_truth(gt_dir, filename)
            if gt_text is None:
                continue
            
            total_with_gt += 1
            decoded_texts = result.get("texts", [])
            
            if decoded_texts:
                # Check if any decoded text matches ground truth
                decoded_text = decoded_texts[0]  # Use first decoded text
                if decoded_text.strip() != gt_text.strip():
                    misread_count += 1
                    misread_cases.append({
                        "file": filename,
                        "ground_truth": gt_text,
                        "decoded": decoded_text
                    })
        
        if total_with_gt > 0:
            misread_rate = misread_count / total_with_gt * 100.0
            misread_details = {
                "total_with_gt": total_with_gt,
                "misread_count": misread_count,
                "misread_cases": misread_cases
            }
    
    return {
        "total_images": total_images,
        "successful_decodes": successful_decodes,
        "decode_success_rate": decode_success_rate,
        "misread_rate": misread_rate,
        "misread_details": misread_details,
        "avg_time_ms": avg_time_ms,
        "total_time_ms": total_time_ms,
        "fps": fps,
        "stage_breakdown": stage_counts
    }


def print_metrics(metrics: Dict[str, Any], method_name: str = "Method") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Metrics dictionary
        method_name: Name of the method for display
    """
    print(f"\n=== {method_name} Evaluation Results ===")
    print(f"Total images: {metrics['total_images']}")
    print(f"Successful decodes: {metrics['successful_decodes']}")
    print(f"Decode success rate: {metrics['decode_success_rate']:.2f}%")
    
    if metrics['misread_rate'] is not None:
        print(f"Misread rate: {metrics['misread_rate']:.2f}%")
        if metrics['misread_details']:
            details = metrics['misread_details']
            print(f"  (Based on {details['total_with_gt']} images with ground truth)")
    else:
        print("Misread rate: N/A (no ground truth provided)")
    
    print(f"Average time per image: {metrics['avg_time_ms']:.1f} ms")
    print(f"Processing speed: {metrics['fps']:.1f} FPS")
    
    # Print stage breakdown
    if metrics['stage_breakdown']:
        print(f"\nStage breakdown:")
        total = metrics['total_images']
        for stage, count in sorted(metrics['stage_breakdown'].items()):
            percentage = count / total * 100.0
            print(f"  {stage}: {count} ({percentage:.1f}%)")


def compare_methods(metrics_list: List[Dict[str, Any]], method_names: List[str]) -> None:
    """
    Compare multiple methods side by side.
    
    Args:
        metrics_list: List of metrics dictionaries
        method_names: List of method names
    """
    print(f"\n=== Method Comparison ===")
    
    # Header
    print(f"{'Metric':<25}", end="")
    for name in method_names:
        print(f"{name:>15}", end="")
    print()
    
    print("-" * (25 + 15 * len(method_names)))
    
    # Decode success rate
    print(f"{'Decode Success (%)':<25}", end="")
    for metrics in metrics_list:
        print(f"{metrics['decode_success_rate']:>14.2f}%", end="")
    print()
    
    # Misread rate (if available)
    if any(m['misread_rate'] is not None for m in metrics_list):
        print(f"{'Misread Rate (%)':<25}", end="")
        for metrics in metrics_list:
            if metrics['misread_rate'] is not None:
                print(f"{metrics['misread_rate']:>14.2f}%", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    
    # Average time
    print(f"{'Avg Time (ms)':<25}", end="")
    for metrics in metrics_list:
        print(f"{metrics['avg_time_ms']:>14.1f}", end="")
    print()
    
    # FPS
    print(f"{'FPS':<25}", end="")
    for metrics in metrics_list:
        print(f"{metrics['fps']:>14.1f}", end="")
    print()


def eval_json(json_path: str, gt_dir: Optional[str] = None, method_name: str = None) -> Dict[str, Any]:
    """
    Evaluate a single JSON results file.
    
    Args:
        json_path: Path to JSON results file
        gt_dir: Directory containing ground truth files
        method_name: Name of the method for display
        
    Returns:
        Metrics dictionary
    """
    # Load results
    results = load_json(json_path)
    
    # Calculate metrics
    metrics = calculate_metrics(results, gt_dir)
    
    # Print results
    if method_name is None:
        method_name = Path(json_path).stem
    
    print_metrics(metrics, method_name)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate QR code decoding results")
    parser.add_argument("--json", required=True, help="JSON results file path")
    parser.add_argument("--gt_dir", help="Ground truth directory (optional)")
    parser.add_argument("--method_name", help="Method name for display")
    
    args = parser.parse_args()
    
    try:
        eval_json(args.json, args.gt_dir, args.method_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
