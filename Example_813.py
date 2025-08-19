import cvxpy as cp

def solution2():
    # Create a 2D decision variable
    x = cp.Variable(2)

    # Define the objective: minimize the norm plus a weighted max function
    objective_function = cp.Minimize(cp.norm(cp.hstack([x, 1])) + 2 * cp.maximum(x[0], x[1], 0))

    # Define the constraints for the optimization problem
    constraints = [
        cp.norm(x, 1) + cp.quad_over_lin(x[0], x[1]) <= 5,
        cp.inv_pos(x[1]) + cp.power(x[0], 4) <= 10,
        x[1] >= 1,
        x[0] >= 0
    ]

    # Set up and solve the optimization problem
    problem = cp.Problem(objective_function, constraints)
    problem.solve()

    # Display the results
    print("Optimized x values:", x.value)
    print("Optimal objective function value:", problem.value)