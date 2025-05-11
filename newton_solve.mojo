from math import sqrt

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

fn function(x: SIMD[DType.float64, 2]) -> SIMD[DType.float64, 2]:
    """Return values of f_1(x, y) and f_2(x, y)."""
    var result = SIMD[DType.float64, 2]()
    result[0] = (x[0] * x[0] + x[1] * x[1]) - 13.0
    result[1] = x[0] * x[0] - 2.0 * x[1] * x[1] + 14.0
    return result

fn jacobian(x: SIMD[DType.float64, 2]) -> Matrix2x2:
    """Return the Jacobian matrix J."""
    return Matrix2x2(
        2.0 * x[0],   # J[0,0] 
        2.0 * x[1],   # J[0,1]
        2.0 * x[0],   # J[1,0]
        -4.0 * x[1]   # J[1,1]
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

fn newtons_method_solve() -> SIMD[DType.float64, 2]:
    """Newton's method using direct solve."""
    # number of iterations to try
    var niters = 20
    
    # tolerance that sets the accuracy of solution
    var tol: Float64 = 1e-6

    # initial guess
    var xk = SIMD[DType.float64, 2](-20.0, 20.0)

    # Newton's method
    for i in range(niters):
        var xk_old = xk
            
        var J = jacobian(xk)
        var step = solve_2x2_system(J, function(xk))
        
        xk = xk - step

        var diff = xk - xk_old
        
        if l2_norm(diff) < tol:
            break

    return xk

fn main():
    # Run Newton's method with direct solve
    var solution = newtons_method_solve()
    print("Solution:", solution) 