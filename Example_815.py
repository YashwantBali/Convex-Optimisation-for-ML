import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def solution4():
    # Generate synthetic data with an outlier
    np.random.seed(314)
    x_vals = np.linspace(-5, 5, 20)
    y_vals = 2 * x_vals + 1 + np.random.randn(20)
    x_vals = np.append(x_vals, 5)
    y_vals = np.append(y_vals, -20)

    # Construct the matrix representation for the regression problem
    A_matrix = np.vstack([x_vals, np.ones(len(x_vals))]).T
    b_vector = y_vals

    # Define optimization variables
    params = cp.Variable(2)

    # Set up the objective: minimize the L1 norm of residuals
    objective_func = cp.Minimize(cp.norm(A_matrix @ params - b_vector, 1))

    # Solve the optimization problem
    problem = cp.Problem(objective_func)
    problem.solve()

    # Display results
    print("Optimized parameters (slope, intercept):", params.value)

    # Plot data points and regression line
    plt.figure(figsize=(6, 6))
    plt.scatter(x_vals, y_vals, marker='*', label="Data Points")
    plt.plot([-6, 6], params.value[0] * np.array([-6, 6]) + params.value[1], 
             label="Robust Regression Line", color='red')
    
    plt.legend()
    plt.title("Robust Linear Regression")
    plt.show(block=False)
