import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def solution3():

    # Defining the points
    A = np.array([[-1, -3, -1, 5, -1], [3, 10, 0, 0, -5]])

    # Defining the variables x and r we need to find
    x = cp.Variable(2)
    r = cp.Variable()

    # Defining the objective function: minimize r
    objective = cp.Minimize(r)

    # Defining the constraints
    constraints = [cp.norm(x - A[:, i]) <= r for i in range(A.shape[1])]

    # Defining the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Results
    print("Optimal value of x:", x.value)
    print("Optimal value of r:", r.value)

    # Plot the points and the Chebyshev circle
    plt.figure(figsize=(6, 6))
    plt.scatter(A[0, :], A[1, :], marker='*', label="Points")
    plt.scatter(x.value[0], x.value[1], marker='D', label="Chebyshev Center")
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(x.value[0] + r.value * np.cos(theta), x.value[1] + r.value * np.sin(theta),
             label="Chebyshev Circle")
    plt.axis("equal")
    plt.legend(loc="upper right")
    plt.title("Chebyshev Center")
    plt.show(block=False)