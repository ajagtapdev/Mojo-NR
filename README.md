# Newton's Method Implementation in Mojo

A high-performance implementation of Newton's method for solving nonlinear systems of equations using Mojo. This project demonstrates different solution techniques and benchmarks their performance.

## Overview

This project implements Newton's method to find the roots of a nonlinear system of equations. Three different solution strategies are implemented and compared:

1. **Matrix Inversion** - Computes the inverse of the Jacobian matrix
2. **Direct Solve** - Solves the linear system directly without computing the inverse
3. **GPU-Accelerated** - Simulates GPU parallelism with reduced computational overhead

The specific system being solved is:
- f₁(x,y) = x² + y² - 13 = 0
- f₂(x,y) = x² - 2y² + 14 = 0

This system has four solutions: (±3, ±2).

## Features

- Multiple Newton's method implementations with performance benchmarking
- Artificial computational work to simulate relative costs of different methods
- Iteration history tracking and convergence analysis
- CSV output for further analysis and visualization
- Support for multiple starting points to analyze convergence behavior

## Prerequisites

- [Mojo SDK](https://www.modular.com/mojo)
- Python 3.x (for visualization scripts)
- Matplotlib (for visualization)
- Pandas (for data processing)

## Running the Code

Compile and run the Mojo code:

```bash
mojo main.mojo
```

This will:
1. Create `data` and `plots` directories
2. Run all three Newton's method implementations
3. Run the GPU-accelerated version with multiple starting points
4. Save results to `data/newton_results.csv`

## Visualizing Results

After running the code, use the Python visualization scripts:

```bash
python visualize.py            # Generates convergence plots
python visualize_gpu_performance.py  # Compares performance metrics
python gpu_benchmark.py        # Detailed GPU benchmarking
```

The visualizations will be saved to the `plots` directory.

## Implementation Details

### Newton's Method

Newton's method is an iterative numerical technique for finding roots of differentiable functions. For systems of equations, it requires:

1. Function evaluation
2. Jacobian matrix computation
3. Solving linear systems

The iterative formula is:
x_{k+1} = x_k - J(x_k)^(-1) * F(x_k)

### Artificial Computational Work

The code includes artificial computational complexity factors to simulate the relative performance differences between methods:
- Matrix inversion: High computational cost (factor: 50000)
- Direct solve: Medium computational cost (factor: 25000)
- GPU acceleration: Low computational cost (factor: 10000)

These simulate the real-world performance characteristics of each approach, especially in higher dimensions.

## Project Structure

- `main.mojo` - Main implementation file
- `data/` - Directory for CSV output
- `plots/` - Directory for visualization output
- Python visualization scripts (not included in this repository)

## Performance Metrics

The code measures:
- Time per iteration
- Total convergence time
- Iterations per second
- Approximate FLOP count

These metrics help compare the efficiency of different numerical methods for solving nonlinear systems. 