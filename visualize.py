import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import matplotlib.animation as animation
from matplotlib.cm import get_cmap
import argparse
import os
import glob

def contour_functions():
    """Create contour plots of the two nonlinear functions."""
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # The functions we're solving
    F1 = X**2 + Y**2 - 13
    F2 = X**2 - 2*Y**2 + 14
    
    return X, Y, F1, F2

def plot_newton_method(file_path, method_name):
    """Visualize the Newton's method convergence."""
    # Load data
    data = pd.read_csv(file_path)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # First subplot: Convergence path
    ax1 = fig.add_subplot(221)
    ax1.set_title(f"{method_name}: Convergence Path")
    
    # Create contour plots
    X, Y, F1, F2 = contour_functions()
    CS1 = ax1.contour(X, Y, F1, levels=[0], colors='blue')
    CS2 = ax1.contour(X, Y, F2, levels=[0], colors='red')
    
    plt.clabel(CS1, inline=True, fontsize=10, fmt='$x^2 + y^2 = 13$')
    plt.clabel(CS2, inline=True, fontsize=10, fmt='$x^2 - 2y^2 = -14$')
    
    # Plot convergence path
    ax1.plot(data['x'], data['y'], 'k-', alpha=0.5)
    ax1.plot(data['x'], data['y'], 'ko', alpha=0.7)
    
    # Mark the starting point
    ax1.plot(data['x'].iloc[0], data['y'].iloc[0], 'go', markersize=10, label='Starting point')
    
    # Mark the final point
    ax1.plot(data['x'].iloc[-1], data['y'].iloc[-1], 'ro', markersize=10, label='Final solution')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    
    # Second subplot: Error vs Iteration
    ax2 = fig.add_subplot(222)
    ax2.set_title(f"{method_name}: Error vs Iteration")
    ax2.semilogy(data['iteration'], data['error'], 'b-o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Error (log scale)')
    ax2.grid(True)
    
    # Third subplot: x vs Iteration
    ax3 = fig.add_subplot(223)
    ax3.set_title(f"{method_name}: x vs Iteration")
    ax3.plot(data['iteration'], data['x'], 'r-o')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('x')
    ax3.grid(True)
    
    # Fourth subplot: y vs Iteration
    ax4 = fig.add_subplot(224)
    ax4.set_title(f"{method_name}: y vs Iteration")
    ax4.plot(data['iteration'], data['y'], 'g-o')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('y')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{method_name.lower().replace(' ', '_')}_plots.png")
    
    return fig, data

def create_animation(data, method_name):
    """Create an animation of the convergence path."""
    # Create contour plots
    X, Y, F1, F2 = contour_functions()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"{method_name}: Convergence Animation")
    
    # Plot the contours
    CS1 = ax.contour(X, Y, F1, levels=[0], colors='blue')
    CS2 = ax.contour(X, Y, F2, levels=[0], colors='red')
    
    plt.clabel(CS1, inline=True, fontsize=10, fmt='$x^2 + y^2 = 13$')
    plt.clabel(CS2, inline=True, fontsize=10, fmt='$x^2 - 2y^2 = -14$')
    
    # Set limits
    margin = 1.0
    x_min, x_max = min(data['x'])-margin, max(data['x'])+margin
    y_min, y_max = min(data['y'])-margin, max(data['y'])+margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Empty line for the convergence path
    line, = ax.plot([], [], 'k-', alpha=0.6)
    
    # Empty scatter for the points
    point = ax.plot([], [], 'ro', markersize=8)[0]
    
    # Text for iteration counter
    iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    # Text for error display
    error_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    
    def init():
        line.set_data([], [])
        point.set_data([], [])
        iteration_text.set_text('')
        error_text.set_text('')
        return line, point, iteration_text, error_text
    
    def animate(i):
        x = data['x'].iloc[:i+1]
        y = data['y'].iloc[:i+1]
        line.set_data(x, y)
        point.set_data(x.iloc[-1], y.iloc[-1])
        iteration_text.set_text(f'Iteration: {i}')
        error_text.set_text(f'Error: {data["error"].iloc[i]:.6f}')
        return line, point, iteration_text, error_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data), interval=300, blit=True)
    
    anim.save(f"{method_name.lower().replace(' ', '_')}_animation.gif", writer='pillow', fps=2)
    
    return anim

def load_csv_safely(filepath):
    """Load a CSV file with error handling"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_basic_plots(csv_file, method_name):
    """Create simple plots from Newton's method CSV results"""
    print(f"Processing {csv_file}...")
    
    # Load data
    df = load_csv_safely(csv_file)
    if df is None:
        return
    
    try:
        # Create figure and subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Newton's Method: {method_name}", fontsize=16)
        
        # Plot trajectory
        axes[0, 0].plot(df['x'], df['y'], 'o-', label='Trajectory')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Convergence Path')
        axes[0, 0].grid(True)
        
        # Plot error
        try:
            axes[0, 1].semilogy(df['iteration'], df['error'], 'o-b')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Error (log scale)')
            axes[0, 1].set_title('Error vs Iteration')
            axes[0, 1].grid(True)
        except Exception as e:
            print(f"Error plotting error: {e}")
            
        # Plot x vs iteration
        axes[1, 0].plot(df['iteration'], df['x'], 'o-r')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('x')
        axes[1, 0].set_title('x vs Iteration')
        axes[1, 0].grid(True)
        
        # Plot y vs iteration
        axes[1, 1].plot(df['iteration'], df['y'], 'o-g')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('y vs Iteration')
        axes[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_path = f"{method_name}_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Created plots: {output_path}")
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"Error creating plots for {csv_file}: {e}")
        plt.close()
        return False

def plot_contours():
    """Plot the contours of the system without relying on data files"""
    print("Creating contour plot...")
    
    # Define system of equations
    def f1(x, y):
        return (x**2 + y**2) - 13.0
    
    def f2(x, y):
        return x**2 - 2*y**2 + 14.0
    
    try:
        # Create grid
        x = np.linspace(-6, 6, 300)
        y = np.linspace(-4, 4, 300)
        X, Y = np.meshgrid(x, y)
        Z1 = f1(X, Y)
        Z2 = f2(X, Y)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        cs1 = plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        cs2 = plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        
        plt.clabel(cs1, inline=1, fontsize=10, fmt='$x^2+y^2=13$')
        plt.clabel(cs2, inline=1, fontsize=10, fmt='$x^2-2y^2=-14$')
        
        # Mark solutions
        solutions = [(3.0, 2.0), (3.0, -2.0), (-3.0, 2.0), (-3.0, -2.0)]
        for sol in solutions:
            plt.plot(sol[0], sol[1], 'k*', markersize=15)
            plt.annotate(f"({sol[0]}, {sol[1]})", xy=sol, xytext=(sol[0]+0.2, sol[1]+0.2))
        
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('System of Equations:\n$x^2+y^2=13$ and $x^2-2y^2=-14$')
        
        plt.savefig("contour_plot.png", dpi=300, bbox_inches='tight')
        print("Created contour plot: contour_plot.png")
        plt.close()
        return True
    except Exception as e:
        print(f"Error creating contour plot: {e}")
        plt.close()
        return False

def main():
    """Main function to process all CSV files and create visualizations"""
    # Process all CSV files
    csv_files = glob.glob("newton_*.csv")
    
    if not csv_files:
        print("No CSV files found. Run the Newton's method code first.")
        return
    
    # Create plots for each method
    for csv_file in csv_files:
        method_name = csv_file.replace(".csv", "").replace("newton_", "")
        create_basic_plots(csv_file, method_name)
    
    # Create contour plot
    plot_contours()
    
    print("Visualization complete. Check the current directory for PNG files.")

if __name__ == "__main__":
    main() 