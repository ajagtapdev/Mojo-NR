import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_convergence_paths(csv_file, plots_dir):
    """Plot the convergence paths for each method and run."""
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Plot settings
    plt.figure(figsize=(12, 10))
    
    # Define marker styles and colors for different methods
    markers = {'newton_inv': 'o', 'newton_solve': 's', 'newton_gpu': 'D'}
    colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'purple', 4: 'orange', 5: 'brown'}
    
    # Prepare the plot
    ax = plt.subplot(111)
    
    # Group by method and run_id
    groups = df.groupby(['method', 'run_id'])
    
    # Exact solutions
    exact_solutions = [(3, 2), (3, -2), (-3, 2), (-3, -2)]
    
    # Plot each group's path
    legend_handles = []
    for name, group in groups:
        method, run_id = name
        marker = markers.get(method, 'x')
        color = colors.get(run_id, 'black')
        
        # Plot the path
        line, = ax.plot(group['x'], group['y'], marker=marker, linestyle='-', 
                      color=color, alpha=0.7, markersize=4,
                      label=f"{method} (run {run_id})")
        
        # Add arrows to show direction
        for i in range(len(group) - 1):
            ax.annotate('', 
                     xy=(group['x'].iloc[i+1], group['y'].iloc[i+1]),
                     xytext=(group['x'].iloc[i], group['y'].iloc[i]),
                     arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.5))
        
        legend_handles.append(line)
    
    # Plot exact solutions
    for i, (x, y) in enumerate(exact_solutions):
        ax.scatter(x, y, color='black', marker='*', s=150, 
                 label=f"Exact solution {i+1}" if i == 0 else None)
        ax.text(x, y+0.5, f"({x}, {y})", ha='center')
    
    # Set plot limits and labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Newton\'s Method Convergence Paths')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right')
    
    # Save the convergence paths plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'convergence_paths.png'), dpi=300)
    plt.close()
    
    # Plot error convergence
    plt.figure(figsize=(12, 8))
    for name, group in groups:
        method, run_id = name
        marker = markers.get(method, 'x')
        color = colors.get(run_id, 'black')
        
        plt.semilogy(group['iteration'], group['error'], marker=marker, 
                   linestyle='-', color=color, alpha=0.7, 
                   label=f"{method} (run {run_id})")
    
    plt.xlabel('Iteration')
    plt.ylabel('Error (log scale)')
    plt.title('Error Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'error_convergence.png'), dpi=300)
    plt.close()
    
    # Create a contour plot of the function
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    
    # Define the functions
    F1 = (X**2 + Y**2) - 13
    F2 = X**2 - 2*Y**2 + 14
    
    # Plot the contours
    plt.figure(figsize=(12, 10))
    cs1 = plt.contour(X, Y, F1, levels=[0], colors='blue', linestyles='-', linewidths=2)
    cs2 = plt.contour(X, Y, F2, levels=[0], colors='red', linestyles='-', linewidths=2)
    
    # Add labels to the contours
    plt.clabel(cs1, inline=True, fontsize=10, fmt='f₁(x,y)=0')
    plt.clabel(cs2, inline=True, fontsize=10, fmt='f₂(x,y)=0')
    
    # Plot the exact solutions
    for x, y in exact_solutions:
        plt.scatter(x, y, color='black', marker='*', s=150)
        plt.text(x, y+0.3, f"({x}, {y})", ha='center')
    
    # Add selected convergence paths
    for name, group in groups:
        method, run_id = name
        if run_id <= 2:  # Only show a few paths to avoid clutter
            plt.plot(group['x'], group['y'], marker='o', linestyle='-', 
                   alpha=0.5, markersize=3, 
                   label=f"{method} (run {run_id})")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Contours and Newton\'s Method Convergence')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'function_contours.png'), dpi=300)
    plt.close()
    
    print(f"Plots saved to {plots_dir} directory")

if __name__ == "__main__":
    csv_file = "data/newton_results.csv"
    plots_dir = "plots"
    
    if os.path.exists(csv_file):
        plot_convergence_paths(csv_file, plots_dir)
    else:
        print(f"Error: CSV file {csv_file} not found.")
        print("Run the main.mojo program first to generate the data.") 