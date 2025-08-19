import cvxpy as cp
import numpy as np

def solution1():
    # Define the given matrix A (3x2) and vector b (3x1)
    A = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([7, 8, 9])

    # Define the optimization variable x (2x1)
    x = cp.Variable(2)

    # Objective function: minimize the squared norm of (Ax - b)
    objective = cp.Minimize(cp.sum_squares(A @ x - b))

    # Solve the optimization problem
    problem = cp.Problem(objective)
    problem.solve()

    # Display results
    print("Optimal value of x:", x.value)
    print("Minimum objective function value:", problem.value)
