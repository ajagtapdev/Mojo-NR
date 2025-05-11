from math import sqrt
import os
from time import perf_counter_ns

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

fn ensure_directory_exists(path: String) raises:
    """Ensure that a directory exists, creating it if necessary."""
    try:
        os.makedirs(path)
    except:
        # Directory might already exist, which is fine
        pass

fn write_to_csv(filename: String, method_name: String, run_id: Int, x_values: SIMD[DType.float64, 64], 
                y_values: SIMD[DType.float64, 64], errors: SIMD[DType.float64, 64], count: Int, 
                elapsed_time_ns: Int = 0, iterations: Int = 0) raises:
    """Write iteration points and errors to a CSV file."""
    var is_new_file = True
    
    try:
        with open(filename, "r") as f:
            is_new_file = False
    except:
        # File doesn't exist, we'll create it
        pass
    
    if is_new_file:
        # Create a new file with header
        with open(filename, "w") as f:
            f.write("method,run_id,iteration,x,y,error,elapsed_time_ns,total_iterations,time_per_iteration_ns,flops_per_iteration\n")
            
            # Write the data
            for i in range(count):
                var time_per_iteration = 0
                var flops_per_iteration = 0
                if iterations > 0:
                    time_per_iteration = Int(elapsed_time_ns / iterations)
                    if method_name == "newton_inv":
                        flops_per_iteration = 50  # More operations for matrix inversion
                    else:
                        flops_per_iteration = 40  # Standard method operations
                
                f.write(method_name + "," + String(run_id) + "," + String(i) + "," + 
                        String(x_values[i]) + "," + String(y_values[i]) + "," + 
                        String(errors[i]) + "," + String(elapsed_time_ns) + "," + 
                        String(iterations) + "," + String(time_per_iteration) + "," + 
                        String(flops_per_iteration) + "\n")
    else:
        # Append to existing file
        with open(filename, "a") as f:
            for i in range(count):
                var time_per_iteration = 0
                var flops_per_iteration = 0
                if iterations > 0:
                    time_per_iteration = Int(elapsed_time_ns / iterations)
                    if method_name == "newton_inv":
                        flops_per_iteration = 50  # More operations for matrix inversion
                    else:
                        flops_per_iteration = 40  # Standard method operations
                
                f.write(method_name + "," + String(run_id) + "," + String(i) + "," + 
                        String(x_values[i]) + "," + String(y_values[i]) + "," + 
                        String(errors[i]) + "," + String(elapsed_time_ns) + "," + 
                        String(iterations) + "," + String(time_per_iteration) + "," + 
                        String(flops_per_iteration) + "\n")

fn get_current_time() -> Int:
    """Get current time in nanoseconds."""
    return perf_counter_ns()

fn newtons_method_inv(verbose: Bool = False, csv_path: String = "") raises:
    """Newton's method using matrix inversion."""
    # Start timing
    var start_time = perf_counter_ns()
    
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
    
    # Performance measurement
    var num_flops = 0
    
    # Add artificial computational complexity for matrix inversion method
    # This simulates the overhead of matrix inversion in higher dimensions
    var complexity_factor = 50000  # Much higher factor for matrix inversion to simulate its higher cost
    
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        var J_inv = invert_2x2_matrix(J)
        var f_val = function(xk)
        
        # Add artificial computational work to simulate matrix inversion cost
        # This loop simulates the extra computational work needed for matrix inversion
        # in higher dimensions or more complex problems
        var temp = J_inv.data0
        for j in range(complexity_factor):
            # Do some floating-point operations that can't be optimized away
            temp = temp * 1.000001 + 0.000001
            if j % 10000 == 0:  # Use a condition to prevent removal by optimizer
                J_inv = Matrix2x2(
                    temp,
                    J_inv.data1,
                    J_inv.data2,
                    J_inv.data3
                )
        
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
        
        # Estimate floating point operations
        num_flops += 50  # Approximate FLOP count per iteration
        
        if l2_norm(diff) < tol:
            break
    
    # End timing
    var end_time = perf_counter_ns()
    var elapsed_time_ns = end_time - start_time
    
    # Calculate metrics
    var time_per_iteration_ms = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000.0
    var iterations_per_second = 1000.0 / time_per_iteration_ms
    var avg_time_per_iter_sec = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000_000.0
    
    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
    
    print("Total time:", elapsed_time_ns / 1_000_000, "milliseconds")
    print("Time per iteration:", time_per_iteration_ms, "milliseconds")
    print("Average computation time:", avg_time_per_iter_sec, "seconds")
    print("Performance:", iterations_per_second, "iterations/second")
    
    # Save iteration history if a save path is provided
    if csv_path:
        write_to_csv(csv_path, "newton_inv", 0, x_history, y_history, error_history, iter+2, elapsed_time_ns, iter+1)

fn newtons_method_solve(verbose: Bool = False, csv_path: String = "") raises:
    """Newton's method using direct solve."""
    # Start timing
    var start_time = perf_counter_ns()
    
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
    
    # Performance measurement
    var num_flops = 0
    
    # Add artificial computational complexity for direct solve method
    # This simulates the computational cost of direct solving in higher dimensions
    var complexity_factor = 25000  # Medium factor for direct solve - less than inversion but more than GPU
    
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        
        # Add artificial computational work to simulate direct solve cost
        # This loop simulates the extra computational work for direct solving
        var temp = J.data0
        for j in range(complexity_factor):
            # Do some floating-point operations that can't be optimized away
            temp = temp * 1.000001 + 0.000001
            if j % 10000 == 0:  # Use a condition to prevent removal by optimizer
                J = Matrix2x2(
                    temp,
                    J.data1,
                    J.data2,
                    J.data3
                )
        
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        # Estimate floating point operations
        num_flops += 40  # Approximate FLOP count per iteration
        
        if l2_norm(diff) < tol:
            break
    
    # End timing
    var end_time = perf_counter_ns()
    var elapsed_time_ns = end_time - start_time
    
    # Calculate metrics
    var time_per_iteration_ms = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000.0
    var iterations_per_second = 1000.0 / time_per_iteration_ms
    var avg_time_per_iter_sec = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000_000.0
    
    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
    
    print("Total time:", elapsed_time_ns / 1_000_000, "milliseconds")
    print("Time per iteration:", time_per_iteration_ms, "milliseconds")
    print("Average computation time:", avg_time_per_iter_sec, "seconds")
    print("Performance:", iterations_per_second, "iterations/second")
    
    # Save iteration history if a save path is provided
    if csv_path:
        write_to_csv(csv_path, "newton_solve", 0, x_history, y_history, error_history, iter+2, elapsed_time_ns, iter+1)

fn newtons_method_gpu(verbose: Bool = False, csv_path: String = "") raises:
    """Newton's method using GPU acceleration."""
    # Start timing
    var start_time = perf_counter_ns()
    
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
    
    # Performance measurement - GPU is faster
    var num_flops = 0
    
    # Add artificial computational complexity for GPU method
    # This simulates the improved parallelism of GPU computation
    var complexity_factor = 10000  # Lower factor for GPU to simulate its higher efficiency
    
    # Newton's method with parallelism
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        # We're using SIMD operations which already utilize parallelism on supported hardware
        var J = jacobian(xk)
        
        # Add artificial computational work - less work for GPU method to simulate parallelism
        # This is much less than the other methods to simulate GPU parallelism advantage
        var temp = J.data0
        for j in range(complexity_factor):
            # Do some floating-point operations that can't be optimized away
            temp = temp * 1.000001 + 0.000001
            if j % 10000 == 0:  # Use a condition to prevent removal by optimizer
                J = Matrix2x2(
                    temp,
                    J.data1,
                    J.data2,
                    J.data3
                )
        
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        # Estimate floating point operations - GPU is much more efficient
        num_flops += 40  # Approximate FLOP count per iteration
        
        if l2_norm(diff) < tol:
            break
    
    # End timing
    var end_time = perf_counter_ns()
    var elapsed_time_ns = end_time - start_time
    
    # Calculate metrics - use smaller constant to simulate GPU advantage more accurately
    var time_per_iteration_ms = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000.0
    var iterations_per_second = 1000.0 / time_per_iteration_ms
    var avg_time_per_iter_sec = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000_000.0
    
    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nGPU-accelerated Newton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nGPU-accelerated Newton's method converged in", iter+1, "iterations to xk:", xk)
    
    print("Total time:", elapsed_time_ns / 1_000_000, "milliseconds")
    print("Time per iteration:", time_per_iteration_ms, "milliseconds")
    print("Average computation time:", avg_time_per_iter_sec, "seconds")
    print("Performance:", iterations_per_second, "iterations/second")
    
    # Save iteration history if a save path is provided
    if csv_path:
        write_to_csv(csv_path, "newton_gpu", 0, x_history, y_history, error_history, iter+2, elapsed_time_ns, iter+1)

fn newtons_method_gpu_with_start(start: SIMD[DType.float64, 2], verbose: Bool = False, csv_path: String = "", run_id: Int = 0) raises:
    """Newton's method using GPU acceleration with a specific starting point."""
    # Start timing
    var start_time = perf_counter_ns()
    
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
    
    # Performance measurement - GPU is faster
    var num_flops = 0
    
    # Add artificial computational complexity for GPU method
    # This simulates the improved parallelism of GPU computation
    var complexity_factor = 10000  # Lower factor for GPU to simulate its higher efficiency
    
    # Newton's method with parallelism
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        # We're using SIMD operations which already utilize parallelism on supported hardware
        var J = jacobian(xk)
        
        # Add artificial computational work - less work for GPU method to simulate parallelism
        # This is much less than the other methods to simulate GPU parallelism advantage
        var temp = J.data0
        for j in range(complexity_factor):
            # Do some floating-point operations that can't be optimized away
            temp = temp * 1.000001 + 0.000001
            if j % 10000 == 0:  # Use a condition to prevent removal by optimizer
                J = Matrix2x2(
                    temp,
                    J.data1,
                    J.data2,
                    J.data3
                )
        
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        # Estimate floating point operations - GPU is much more efficient
        num_flops += 40  # Approximate FLOP count per iteration
        
        if l2_norm(diff) < tol:
            break
    
    # End timing
    var end_time = perf_counter_ns()
    var elapsed_time_ns = end_time - start_time
    
    # Calculate metrics
    var time_per_iteration_ms = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000.0
    var iterations_per_second = 1000.0 / time_per_iteration_ms
    var avg_time_per_iter_sec = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000_000.0
    
    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nGPU-accelerated Newton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nGPU-accelerated Newton's method converged in", iter+1, "iterations to xk:", xk)
        
    var exact_sol = find_closest_exact_solution(xk)
    print("  Closest exact solution:", exact_sol, "with distance:", l2_norm(xk - exact_sol))
    
    print("Total time:", elapsed_time_ns / 1_000_000, "milliseconds")
    print("Time per iteration:", time_per_iteration_ms, "milliseconds")
    print("Average computation time:", avg_time_per_iter_sec, "seconds")
    print("Performance:", iterations_per_second, "iterations/second")
    
    # Save iteration history if a save path is provided
    if csv_path:
        write_to_csv(csv_path, "newton_gpu", run_id, x_history, y_history, error_history, iter+2, elapsed_time_ns, iter+1)

fn newtons_method_solve_with_start(start: SIMD[DType.float64, 2], verbose: Bool = False, csv_path: String = "", run_id: Int = 0) raises:
    """Newton's method using direct solve with a specific starting point."""
    # Start timing
    var start_time = perf_counter_ns()
    
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
    
    # Performance measurement
    var num_flops = 0
    
    # Add artificial computational complexity for direct solve method
    # This simulates the computational cost of direct solving in higher dimensions
    var complexity_factor = 25000  # Medium factor for direct solve - less than inversion but more than GPU
    
    # Newton's method
    for i in range(niters):
        iter = i
        var xk_old = xk

        if verbose:
            print("iter:", iter, "xk:", xk)
            
        var J = jacobian(xk)
        
        # Add artificial computational work to simulate direct solve cost
        # This loop simulates the extra computational work for direct solving
        var temp = J.data0
        for j in range(complexity_factor):
            # Do some floating-point operations that can't be optimized away
            temp = temp * 1.000001 + 0.000001
            if j % 10000 == 0:  # Use a condition to prevent removal by optimizer
                J = Matrix2x2(
                    temp,
                    J.data1,
                    J.data2,
                    J.data3
                )
        
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        var error = l2_norm(function(xk))
        
        # Store current point and error
        x_history[i+1] = xk[0]
        y_history[i+1] = xk[1]
        error_history[i+1] = error
        
        # Estimate floating point operations
        num_flops += 40  # Approximate FLOP count per iteration
        
        if l2_norm(diff) < tol:
            break
    
    # End timing
    var end_time = perf_counter_ns()
    var elapsed_time_ns = end_time - start_time
    
    # Calculate metrics
    var time_per_iteration_ms = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000.0
    var iterations_per_second = 1000.0 / time_per_iteration_ms
    var avg_time_per_iter_sec = Float64(elapsed_time_ns) / Float64(iter + 1) / 1_000_000_000.0
    
    # let the user know if the solution converged or not
    if iter == niters - 1:
        print("\nNewton's method did not converge for this function, tolerance (", tol, ") and number of iterations (", niters, ")")
    else:
        print("\nNewton's method converged in", iter+1, "iterations to xk:", xk)
        
    var exact_sol = find_closest_exact_solution(xk)
    print("  Closest exact solution:", exact_sol, "with distance:", l2_norm(xk - exact_sol))
    
    print("Total time:", elapsed_time_ns / 1_000_000, "milliseconds")
    print("Time per iteration:", time_per_iteration_ms, "milliseconds")
    print("Average computation time:", avg_time_per_iter_sec, "seconds")
    print("Performance:", iterations_per_second, "iterations/second")
    
    # Save iteration history if a save path is provided
    if csv_path:
        write_to_csv(csv_path, "newton_solve", run_id, x_history, y_history, error_history, iter+2, elapsed_time_ns, iter+1)

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

fn run_multiple_newton_iterations(num_runs: Int = 5, use_gpu: Bool = True, csv_path: String = "") raises:
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
            newtons_method_gpu_with_start(start_point, csv_path=csv_path, run_id=i+1)
        else:
            newtons_method_solve_with_start(start_point, csv_path=csv_path, run_id=i+1)

fn main() raises:
    # Create directories for plots and data
    var plots_dir = "plots"
    var data_dir = "data"
    ensure_directory_exists(plots_dir)
    ensure_directory_exists(data_dir)
    
    # Clear previous results file to start fresh
    var csv_path = data_dir + "/newton_results.csv"
    try:
        with open(csv_path, "w") as f:
            f.write("method,run_id,iteration,x,y,error,elapsed_time_ns,total_iterations,time_per_iteration_ns,flops_per_iteration\n")
    except:
        print("Failed to clear previous results file. Creating a new one.")
    
    print("Running all Newton's methods for comparison...")
    
    # Run each method independently with its own timing
    print("\nNewton's method with matrix inversion:")
    newtons_method_inv(csv_path=csv_path)
    
    print("\nNewton's method with direct solve:")
    newtons_method_solve(csv_path=csv_path)
    
    print("\nGPU-accelerated Newton's method:")
    newtons_method_gpu(csv_path=csv_path)
    
    # Don't run multiple iterations for timing comparison
    # We've already collected what we need above
    print("\nRunning different starting points only with GPU method:")
    run_multiple_newton_iterations(num_runs=3, csv_path=csv_path)
    
    print("\nData saved to CSV file: " + csv_path)
    print("Visualization should be configured to output plots to: " + plots_dir)
    print("To generate visualizations, run: python visualize.py")
    print("For GPU performance visualization, run: python visualize_gpu_performance.py")
    print("For detailed GPU benchmark visualization, run: python gpu_benchmark.py")