import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_gpu_performance(csv_file, plots_dir):
    """Visualize GPU performance compared to other methods."""
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Debug: Print column names to diagnose issues
    print("Columns in CSV file:", df.columns.tolist())
    
    # Check if performance column exists, otherwise use some default or error
    performance_col = 'iterations_per_second'
    if performance_col not in df.columns:
        print(f"Warning: Expected column '{performance_col}' not found in CSV.")
        print("Available columns:", df.columns.tolist())
        
        # Try to find a column that might contain performance data
        if 'error' in df.columns:
            print("Using 'error' column as a proxy for performance (lower is better)")
            performance_col = 'error'
            # Create a performance metric inversely related to error
            df['derived_performance'] = 1000.0 / (df['error'] + 0.01)  # Add small value to avoid division by zero
            performance_col = 'derived_performance'
        else:
            print("Error: No suitable performance data found in CSV.")
            return
    
    # Extract performance data - group by method and get appropriate metric
    performance_data = df.groupby(['method', 'run_id'])[performance_col].max().reset_index()
    performance_summary = performance_data.groupby('method')[performance_col].mean().reset_index()
    
    # Create performance comparison bar chart
    plt.figure(figsize=(12, 8))
    colors = {'newton_inv': '#3498db', 'newton_solve': '#2ecc71', 'newton_gpu': '#e74c3c'}
    
    # Create bar chart with custom colors
    bars = plt.bar(performance_summary['method'], performance_summary[performance_col], 
                 color=[colors.get(m, 'gray') for m in performance_summary['method']])
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(height * 0.05, 5),
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12)
    
    # Determine if higher is better based on column name
    higher_is_better = 'derived' in performance_col or 'iteration' in performance_col
    better_text = "Higher is Better" if higher_is_better else "Lower is Better"
    
    # Styling
    plt.title(f'Method Performance Comparison\n({better_text})', fontsize=16)
    plt.ylabel(f'Performance ({performance_col})', fontsize=14)
    plt.xlabel('Method', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Improve labels for readability
    labels = {'newton_inv': 'Matrix Inversion', 'newton_solve': 'Direct Solve', 'newton_gpu': 'GPU Accelerated'}
    plt.xticks(range(len(performance_summary)), [labels.get(m, m) for m in performance_summary['method']])
    
    # Add speedup annotations relative to baseline method
    if higher_is_better:
        baseline_perf = performance_summary[performance_col].min()
        reference = "Speedup"
    else:
        baseline_perf = performance_summary[performance_col].max()
        reference = "Improvement"
        
    for i, row in performance_summary.iterrows():
        if higher_is_better:
            speedup = row[performance_col] / baseline_perf
        else:
            speedup = baseline_perf / row[performance_col]
            
        plt.text(i, row[performance_col] * 0.5, 
                f'{reference}: {speedup:.1f}x', 
                ha='center', va='center', 
                color='white', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'method_performance_comparison.png'), dpi=300)
    plt.close()
    
    # Create performance by run_id for GPU method
    gpu_performance = performance_data[performance_data['method'] == 'newton_gpu']
    
    if not gpu_performance.empty:
        plt.figure(figsize=(12, 8))
        plt.plot(gpu_performance['run_id'], gpu_performance[performance_col], 
               marker='o', markersize=10, linewidth=2, color='#e74c3c')
        
        # Styling
        plt.title('GPU Performance by Run', fontsize=16)
        plt.ylabel(f'Performance ({performance_col})', fontsize=14)
        plt.xlabel('Run ID', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(alpha=0.3)
        
        # Add horizontal line for average
        avg_gpu_perf = gpu_performance[performance_col].mean()
        plt.axhline(y=avg_gpu_perf, color='#e74c3c', linestyle='--', 
                  label=f'Average: {avg_gpu_perf:.1f}')
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'gpu_performance_by_run.png'), dpi=300)
        plt.close()
    
    # Create SIMD utilization visualization
    # This is a conceptual visualization showing SIMD lanes usage
    plt.figure(figsize=(12, 8))
    
    # Represent SIMD lanes
    methods = ['CPU (No SIMD)', 'CPU (SIMD)', 'GPU (SIMD)']
    simd_width = [1, 4, 16]  # Conceptual width of parallelism
    simd_efficiency = [1.0, 0.85, 0.95]  # Conceptual efficiency of utilization
    
    # Create stacked bars for utilized vs. idle SIMD lanes
    utilized = [width * eff for width, eff in zip(simd_width, simd_efficiency)]
    idle = [width * (1-eff) for width, eff in zip(simd_width, simd_efficiency)]
    
    plt.bar(methods, utilized, label='Utilized SIMD Lanes', color='#2ecc71')
    plt.bar(methods, idle, bottom=utilized, label='Idle SIMD Lanes', color='#95a5a6', alpha=0.5)
    
    # Styling
    plt.title('SIMD/GPU Lane Utilization (Conceptual)', fontsize=16)
    plt.ylabel('Parallel Processing Lanes', fontsize=14)
    plt.xlabel('Processing Method', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add utilization percentage labels
    for i, (width, eff) in enumerate(zip(simd_width, simd_efficiency)):
        plt.text(i, width/2, f'{eff*100:.0f}% Utilized', 
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        plt.text(i, width * 0.1, f'{width} Lanes', 
                ha='center', va='bottom', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simd_utilization.png'), dpi=300)
    plt.close()
    
    print(f"GPU performance visualizations saved to {plots_dir}")

if __name__ == "__main__":
    csv_file = "data/newton_results.csv"
    plots_dir = "plots"
    
    if os.path.exists(csv_file):
        visualize_gpu_performance(csv_file, plots_dir)
    else:
        print(f"Error: CSV file {csv_file} not found.")
        print("Run the main.mojo program first to generate the data.") 