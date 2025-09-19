"""
Summarize and compare results from multiple methods and datasets.
Generates tables and visualizations for paper publication.
"""

import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found, skipping advanced visualizations")
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.io import load_json, ensure_dir
from eval.metrics import calculate_metrics


def parse_items_arg(items_str: str) -> List[Tuple[str, str]]:
    """
    Parse items argument in format: path1:name1,path2:name2,...
    
    Args:
        items_str: Comma-separated list of path:name pairs
        
    Returns:
        List of (path, name) tuples
    """
    items = []
    for item in items_str.split(','):
        item = item.strip()
        # 处理Windows路径：找到最后一个冒号作为分隔符
        # 跳过Windows驱动器字母后的冒号 (如 C:)
        colon_pos = -1
        for i in range(len(item) - 1, -1, -1):
            if item[i] == ':':
                # 检查是否是Windows驱动器字母
                if i == 1 and len(item) > 2 and item[0].isalpha():
                    continue  # 跳过驱动器字母的冒号
                colon_pos = i
                break
        
        if colon_pos > 1:  # 找到有效的分隔冒号
            path = item[:colon_pos].strip()
            name = item[colon_pos + 1:].strip()
            items.append((path, name))
        else:
            # Use filename as name if no name provided
            path = item.strip()
            name = Path(path).stem
            items.append((path, name))
    
    return items


def create_summary_table(results_data: List[Dict[str, Any]], method_names: List[str]) -> pd.DataFrame:
    """
    Create summary table from multiple method results.
    
    Args:
        results_data: List of metrics dictionaries
        method_names: List of method names
        
    Returns:
        Pandas DataFrame with summary table
    """
    summary_data = []
    
    for i, (metrics, method_name) in enumerate(zip(results_data, method_names)):
        row = {
            'Method': method_name,
            'Total Images': metrics['total_images'],
            'Successful Decodes': metrics['successful_decodes'],
            'Decode Success Rate (%)': round(metrics['decode_success_rate'], 2),
            'Avg Time (ms)': round(metrics['avg_time_ms'], 1),
            'FPS': round(metrics['fps'], 1)
        }
        
        if metrics['misread_rate'] is not None:
            row['Misread Rate (%)'] = round(metrics['misread_rate'], 2)
        else:
            row['Misread Rate (%)'] = 'N/A'
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def create_stage_breakdown_table(results_data: List[Dict[str, Any]], method_names: List[str]) -> pd.DataFrame:
    """
    Create stage breakdown table for methods that support it.
    
    Args:
        results_data: List of metrics dictionaries
        method_names: List of method names
        
    Returns:
        Pandas DataFrame with stage breakdown
    """
    # Collect all unique stages
    all_stages = set()
    for metrics in results_data:
        if 'stage_breakdown' in metrics:
            all_stages.update(metrics['stage_breakdown'].keys())
    
    all_stages = sorted(all_stages)
    
    stage_data = []
    for metrics, method_name in zip(results_data, method_names):
        if 'stage_breakdown' not in metrics or not metrics['stage_breakdown']:
            continue
        
        total_images = metrics['total_images']
        row = {'Method': method_name}
        
        for stage in all_stages:
            count = metrics['stage_breakdown'].get(stage, 0)
            percentage = count / total_images * 100.0 if total_images > 0 else 0.0
            row[f'{stage} (%)'] = round(percentage, 1)
        
        stage_data.append(row)
    
    return pd.DataFrame(stage_data) if stage_data else None


def plot_comparison_charts(results_data: List[Dict[str, Any]], method_names: List[str], output_dir: str) -> None:
    """
    Create comparison charts and save them.
    
    Args:
        results_data: List of metrics dictionaries
        method_names: List of method names
        output_dir: Output directory for plots
    """
    ensure_dir(output_dir)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # 1. Success Rate vs Speed comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate bar chart
    success_rates = [m['decode_success_rate'] for m in results_data]
    bars1 = ax1.bar(method_names, success_rates)
    ax1.set_ylabel('Decode Success Rate (%)')
    ax1.set_title('Decode Success Rate Comparison')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Speed bar chart
    fps_values = [m['fps'] for m in results_data]
    bars2 = ax2.bar(method_names, fps_values)
    ax2.set_ylabel('Processing Speed (FPS)')
    ax2.set_title('Processing Speed Comparison')
    
    # Add value labels on bars
    for bar, fps in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{fps:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Pareto plot: Success Rate vs Average Time
    fig, ax = plt.subplots(figsize=(8, 6))
    
    avg_times = [m['avg_time_ms'] for m in results_data]
    
    scatter = ax.scatter(avg_times, success_rates, s=100, alpha=0.7)
    
    # Add method labels with overlap detection
    positions = list(zip(avg_times, success_rates))
    
    for i, name in enumerate(method_names):
        x, y = positions[i]
        
        # Check for overlapping positions (within 5% tolerance)
        overlapping = False
        for j, (other_x, other_y) in enumerate(positions):
            if i != j and abs(x - other_x) < max(avg_times) * 0.05 and abs(y - other_y) < 5:
                overlapping = True
                break
        
        # Adjust label position if overlapping
        if overlapping and name in ['Baseline_A', 'Baseline_B']:
            # Move label up and left for overlapping BaseA/BaseB
            offset = (-15, 10) if name == 'Baseline_B' else (-10, 20)
        else:
            offset = (10, 10)
            
        ax.annotate(name, (x, y), xytext=offset, textcoords='offset points')
    
    ax.set_xlabel('Average Time per Image (ms)')
    ax.set_ylabel('Decode Success Rate (%)')
    ax.set_title('Success Rate vs Processing Time (Pareto Analysis)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'pareto_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Stage breakdown (if available)
    stage_df = create_stage_breakdown_table(results_data, method_names)
    if stage_df is not None and len(stage_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for stacked bar chart
        stage_cols = [col for col in stage_df.columns if col != 'Method' and '(%)' in col]
        stage_names = [col.replace(' (%)', '') for col in stage_cols]
        
        bottom = None
        colors = plt.cm.Set3(range(len(stage_cols)))
        
        for i, col in enumerate(stage_cols):
            values = stage_df[col].values
            ax.bar(stage_df['Method'], values, bottom=bottom, 
                  label=stage_names[i], color=colors[i])
            
            if bottom is None:
                bottom = values
            else:
                bottom += values
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Stage Breakdown by Method')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'stage_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Charts saved to: {output_dir}")


def summarize_results(items: List[Tuple[str, str]], output_csv: str, gt_dir: str = None, 
                     create_plots: bool = True) -> None:
    """
    Summarize results from multiple JSON files.
    
    Args:
        items: List of (json_path, method_name) tuples
        output_csv: Output CSV file path
        gt_dir: Ground truth directory (optional)
        create_plots: Whether to create visualization plots
    """
    results_data = []
    method_names = []
    
    print("Loading and analyzing results...")
    
    for json_path, method_name in items:
        print(f"Processing: {method_name} ({json_path})")
        
        try:
            # Load results
            results = load_json(json_path)
            
            # Calculate metrics
            metrics = calculate_metrics(results, gt_dir)
            
            results_data.append(metrics)
            method_names.append(method_name)
            
            print(f"  -> {metrics['successful_decodes']}/{metrics['total_images']} successful "
                  f"({metrics['decode_success_rate']:.1f}%)")
            
        except Exception as e:
            print(f"  -> Error: {e}")
            continue
    
    if not results_data:
        print("No valid results found!")
        return
    
    # Create summary table
    summary_df = create_summary_table(results_data, method_names)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    
    print(f"\nSummary table saved to: {output_csv}")
    print("\n" + "="*80)
    print(summary_df.to_string(index=False))
    
    # Create stage breakdown table if applicable
    stage_df = create_stage_breakdown_table(results_data, method_names)
    if stage_df is not None:
        stage_csv = output_path.parent / f"{output_path.stem}_stages.csv"
        stage_df.to_csv(stage_csv, index=False)
        print(f"\nStage breakdown saved to: {stage_csv}")
        print("\n" + "="*50)
        print(stage_df.to_string(index=False))
    
    # Create plots
    if create_plots:
        try:
            plot_dir = output_path.parent / "plots"
            plot_comparison_charts(results_data, method_names, str(plot_dir))
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare QR decoding results")
    parser.add_argument("--items", required=True, 
                       help="Comma-separated list of json_path:method_name pairs")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    parser.add_argument("--gt_dir", help="Ground truth directory (optional)")
    parser.add_argument("--no_plots", action="store_true", help="Skip creating plots")
    
    args = parser.parse_args()
    
    try:
        items = parse_items_arg(args.items)
        summarize_results(items, args.out, args.gt_dir, not args.no_plots)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
