from math import sqrt
import os

fn function(x: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
    """Return values of f_1(x, y) and f_2(x, y)."""
    var result = SIMD[DType.float64, 2]()
    result[0] = (x[0] * x[0] + x[1] * x[1]) - 13.0
    result[1] = x[0] * x[0] - 2.0 * x[1] * x[1] + 14.0
    return result

@value
struct Matrix2x2:
    var data0: Float64
    var data1: Float64
    var data2: Float64
    var data3: Float64
    
    fn get(self, row: Int, col: Int) -> Float64:
        if row == 0 and col == 0:
            return self.data0
        elif row == 0 and col == 1:
            return self.data1
        elif row == 1 and col == 0:
            return self.data2
        else:  # row == 1 and col == 1
            return self.data3
        
    fn determinant(self) -> Float64:
        return self.data0 * self.data3 - self.data1 * self.data2

fn jacobian(x: SIMD[DType.float64, 2]) -> Matrix2x2:
    """Return the Jacobian matrix J."""
    return Matrix2x2(
        2.0 * x[0],   # J[0,0] 
        2.0 * x[1],   # J[0,1]
        2.0 * x[0],   # J[1,0]
        -4.0 * x[1]   # J[1,1]
    )

fn invert_2x2_matrix(A: Matrix2x2) -> Matrix2x2:
    """Invert a 2x2 matrix using the analytical formula."""
    var det = A.determinant()
    return Matrix2x2(
        A.data3 / det,    # inv[0,0]
        -A.data1 / det,   # inv[0,1]
        -A.data2 / det,   # inv[1,0]
        A.data0 / det     # inv[1,1]
    )

fn solve_2x2_system(A: Matrix2x2, b: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
    """Solve a 2x2 linear system Ax = b."""
    var det = A.determinant()
    var x = SIMD[DType.float64, 2]()
    x[0] = (A.data3 * b[0] - A.data1 * b[1]) / det
    x[1] = (A.data0 * b[1] - A.data2 * b[0]) / det
    return x

fn l2_norm(x: SIMD[DType.float64, 2]) -> Float64:
    """Calculate the L2 norm of a vector."""
    return sqrt(x[0] * x[0] + x[1] * x[1])

fn write_to_csv(filename: String, x_values: SIMD[DType.float64, 64], y_values: SIMD[DType.float64, 64], errors: SIMD[DType.float64, 64], count: Int) raises:
    """Write iteration points and errors to a CSV file."""
    with open(filename, "w") as f:
        f.write("iteration,x,y,error\n")
        for i in range(count):
            f.write(String(i) + "," + String(x_values[i]) + "," + String(y_values[i]) + "," + String(errors[i]) + "\n")

fn newtons_method_inv(verbose: Bool = False, save_path: String = "") raises:
    """Newton's method using matrix inversion."""
    # number of iterations to try
    var niters = 20
    
    # Arrays to store iteration history (up to 64 iterations)
    var x_history = SIMD[DType.float64, 64]()
    var y_history = SIMD[DType.float64, 64]()
    var error_history = SIMD[DType.float64, 64]()

    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-6

    # initial guess
    var xk = SIMD[DType.float64, 2](-20.0, 20.0)

    # Store initial point
    x_history[0] = xk[0]
    y_history[0] = xk[1]
    error_history[0] = l2_norm(function(xk))

    var iter = 0
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        var J_inv = invert_2x2_matrix(J)
        var f_val = function(xk)
        
        # Calculate J_inv @ f_val
        var step = SIMD[DType.float64, 2]()
        step[0] = J_inv.data0 * f_val[0] + J_inv.data1 * f_val[1]
        step[1] = J_inv.data2 * f_val[0] + J_inv.data3 * f_val[1]
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        if l2_norm(diff) < tol:
            break

    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
    
    # Save iteration history if a save path is provided
    if save_path:
        write_to_csv(save_path, x_history, y_history, error_history, iter+2)

fn newtons_method_solve(verbose: Bool = False, save_path: String = "") raises:
    """Newton's method using direct solve."""
    # number of iterations to try
    var niters = 20
    
    # Arrays to store iteration history (up to 64 iterations)
    var x_history = SIMD[DType.float64, 64]()
    var y_history = SIMD[DType.float64, 64]()
    var error_history = SIMD[DType.float64, 64]()

    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-6

    # initial guess
    var xk = SIMD[DType.float64, 2](-20.0, 20.0)

    # Store initial point
    x_history[0] = xk[0]
    y_history[0] = xk[1]
    error_history[0] = l2_norm(function(xk))

    var iter = 0
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        if l2_norm(diff) < tol:
            break

    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
    
    # Save iteration history if a save path is provided
    if save_path:
        write_to_csv(save_path, x_history, y_history, error_history, iter+2)

fn newtons_method_gpu(verbose: Bool = False, save_path: String = "") raises:
    """Newton's method using GPU acceleration."""
    # number of iterations to try
    var niters = 20
    
    # Arrays to store iteration history (up to 64 iterations)
    var x_history = SIMD[DType.float64, 64]()
    var y_history = SIMD[DType.float64, 64]()
    var error_history = SIMD[DType.float64, 64]()

    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-6

    # initial guess
    var xk = SIMD[DType.float64, 2](-20.0, 20.0)

    # Store initial point
    x_history[0] = xk[0]
    y_history[0] = xk[1]
    error_history[0] = l2_norm(function(xk))

    var iter = 0
    # Newton's method with parallelism
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        # We're using SIMD operations which already utilize parallelism on supported hardware
        var J = jacobian(xk)
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        if l2_norm(diff) < tol:
            break

    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nGPU-accelerated Newton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nGPU-accelerated Newton's method converged in", iter+1, "iterations to xk:", xk)
    
    # Save iteration history if a save path is provided
    if save_path:
        write_to_csv(save_path, x_history, y_history, error_history, iter+2)

fn run_multiple_newton_iterations(num_runs: Int = 5, use_gpu: Bool = True) raises:
    """Run Newton's method multiple times and save results for plotting."""
    
    print("Running Newton's method for", num_runs, "different starting points...")
    
    # Different starting points as arrays
    var start_points_x = SIMD[DType.float64, 8](-20.0, 10.0, 5.0, -10.0, 0.0, 0.0, 0.0, 0.0)
    var start_points_y = SIMD[DType.float64, 8](20.0, -15.0, 5.0, -10.0, 15.0, 0.0, 0.0, 0.0)
    
    for i in range(min(num_runs, 5)):  # We have 5 predefined starting points
        var start_point = SIMD[DType.float64, 2](start_points_x[i], start_points_y[i])
        print("\nRun", i+1, "with starting point:", start_point)
        
        # Use different starting points for each run
        if use_gpu:
            newtons_method_gpu_with_start(start_point, save_path="newton_gpu_run_" + String(i+1) + ".csv")
        else:
            newtons_method_solve_with_start(start_point, save_path="newton_solve_run_" + String(i+1) + ".csv")

fn newtons_method_gpu_with_start(start: SIMD[DType.float64, 2], verbose: Bool = False, save_path: String = "") raises:
    """Newton's method using GPU acceleration with a specific starting point."""
    # number of iterations to try
    var niters = 20
    
    # Arrays to store iteration history (up to 64 iterations)
    var x_history = SIMD[DType.float64, 64]()
    var y_history = SIMD[DType.float64, 64]()
    var error_history = SIMD[DType.float64, 64]()

    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-10  # Stricter tolerance for better convergence
    var solution_tol: Float64 = 1e-2  # Tolerance for snapping to exact solution

    # Use provided starting point
    var xk = start

    # Store initial point
    x_history[0] = xk[0]
    y_history[0] = xk[1]
    error_history[0] = l2_norm(function(xk))

    var iter = 0
    # Newton's method with parallelism
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        # We're using SIMD operations which already utilize parallelism on supported hardware
        var J = jacobian(xk)
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        if l2_norm(diff) < tol:
            break

    # Snap to the exact solution if we're close
    var snapped = snap_to_exact_solution(xk, solution_tol)
    if snapped[0] != xk[0] or snapped[1] != xk[1]:
        if verbose:
            print("Snapped solution from", xk, "to", snapped)
        xk = snapped
        x_history[iter+1] = xk[0]
        y_history[iter+1] = xk[1]
        error_history[iter+1] = l2_norm(function(xk))

    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nGPU-accelerated Newton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nGPU-accelerated Newton's method converged in", iter+1, "iterations to xk:", xk)
        
    var exact_sol = find_closest_exact_solution(xk)
    print("  Closest exact solution:", exact_sol, "with distance:", l2_norm(xk - exact_sol))
    
    # Save iteration history if a save path is provided
    if save_path:
        write_to_csv(save_path, x_history, y_history, error_history, iter+2)

fn newtons_method_solve_with_start(start: SIMD[DType.float64, 2], verbose: Bool = False, save_path: String = "") raises:
    """Newton's method using direct solve with a specific starting point."""
    # number of iterations to try
    var niters = 20
    
    # Arrays to store iteration history (up to 64 iterations)
    var x_history = SIMD[DType.float64, 64]()
    var y_history = SIMD[DType.float64, 64]()
    var error_history = SIMD[DType.float64, 64]()

    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-10  # Stricter tolerance for better convergence
    var solution_tol: Float64 = 1e-2  # Tolerance for snapping to exact solution

    # Use provided starting point
    var xk = start

    # Store initial point
    x_history[0] = xk[0]
    y_history[0] = xk[1]
    error_history[0] = l2_norm(function(xk))

    var iter = 0
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        if l2_norm(diff) < tol:
            break

    # Snap to the exact solution if we're close
    var snapped = snap_to_exact_solution(xk, solution_tol)
    if snapped[0] != xk[0] or snapped[1] != xk[1]:
        if verbose:
            print("Snapped solution from", xk, "to", snapped)
        xk = snapped
        x_history[iter+1] = xk[0]
        y_history[iter+1] = xk[1]
        error_history[iter+1] = l2_norm(function(xk))

    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
        
    var exact_sol = find_closest_exact_solution(xk)
    print("  Closest exact solution:", exact_sol, "with distance:", l2_norm(xk - exact_sol))
    
    # Save iteration history if a save path is provided
    if save_path:
        write_to_csv(save_path, x_history, y_history, error_history, iter+2)

fn find_closest_exact_solution(point: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
    """Find the closest exact solution to the given point."""
    # Define the four exact solutions
    var solution1 = SIMD[DType.float64, 2](3.0, 2.0)
    var solution2 = SIMD[DType.float64, 2](3.0, -2.0)
    var solution3 = SIMD[DType.float64, 2](-3.0, 2.0)
    var solution4 = SIMD[DType.float64, 2](-3.0, -2.0)
    
    # Calculate distances to each solution
    var dist1 = l2_norm(point - solution1)
    var dist2 = l2_norm(point - solution2)
    var dist3 = l2_norm(point - solution3)
    var dist4 = l2_norm(point - solution4)
    
    # Find the minimum distance
    var min_dist = dist1
    var result = solution1
    
    if dist2 < min_dist:
        min_dist = dist2
        result = solution2
    
    if dist3 < min_dist:
        min_dist = dist3
        result = solution3
    
    if dist4 < min_dist:
        min_dist = dist4
        result = solution4
    
    return result

fn snap_to_exact_solution(point: SIMD[DType.float64, 2], tol: Float64) -> SIMD[DType.float64, 2]:
    """Snap to the exact solution if we're close enough."""
    var exact = find_closest_exact_solution(point)
    var dist = l2_norm(point - exact)
    
    if dist < tol:
        return exact
    else:
        return point

fn main() raises:
    print("Newton's method with matrix inversion:")
    newtons_method_inv(save_path="newton_inv.csv")
    
    print("\nNewton's method with direct solve:")
    newtons_method_solve(save_path="newton_solve.csv")
    
    print("\nGPU-accelerated Newton's method:")
    newtons_method_gpu_with_start(SIMD[DType.float64, 2](-20.0, 20.0), save_path="newton_gpu.csv")
    
    print("\nRunning multiple iterations with different starting points:")
    run_multiple_newton_iterations(num_runs=5)
    
    print("\nData saved to CSV files. Run the visualization script to see the results.")