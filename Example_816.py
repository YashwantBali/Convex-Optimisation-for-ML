import cvxpy as cp
import numpy as np

def solution5():
    # Define the symmetric matrix A and vector b
    A = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 3]])
    b = np.array([0.5, 1, -0.5])

    # Compute eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    transformed_b = eigenvectors.T @ b

    # Define the optimization variable
    z = cp.Variable(3)

    # Formulate the objective function
    objective_func = cp.Minimize(eigenvalues.T @ z - 2 * cp.abs(transformed_b).T @ cp.sqrt(z))

    # Define constraints
    constraints = [cp.sum(z) <= 1, z >= 0]

    # Solve the optimization problem
    problem = cp.Problem(objective_func, constraints)
    problem.solve()

    # Compute the optimal solution
    y_opt = -np.sign(transformed_b) * np.sqrt(z.value)
    x_opt = eigenvectors @ y_opt

    # Print the final result
    print(x_opt)
