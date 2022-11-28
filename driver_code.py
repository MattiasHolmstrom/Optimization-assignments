
import numpy as np
from lmq_splitted import LevenbergMarquardt


def main():

    # TEST CASE 1 (given in the assignment)

    print("Running Test case 1 (given in assignment)")
    print("Function is y(t) = x_1 * e^(x_2 * t)")

    # Define values of t, y_t and x0
    t = np.arange(0.5, 4.5, 0.5)
    y_t = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
    x0 = [1, 1]

    # Define test function
    def test_function_1(x_k, x, y):
        Fx = y - x_k[0] * np.exp(x_k[1] * x)
        return Fx

    # Initialize optimizer without gradient argument
    optimizer_1 = LevenbergMarquardt(
        func=test_function_1,
        grad=None,
        tol=1e-3,
        lambda_=1,
        alpha=1,
        max_iter=1000,
        plot_sol=True,
        plot_conv=True,
    )

    # Run optimizer
    optimizer_1.minimize(t, y_t, x0)

    # Define gradient function
    def test_gradient_1(x, x_k):
        grad_Fx = np.array([-np.exp(x_k[1] * x), -x * x_k[0] * np.exp(x_k[1] * x)])
        return grad_Fx

    # Initialize optimizer without gradient argument
    optimizer_2 = LevenbergMarquardt(
        func=test_function_1,
        grad=test_gradient_1,
        tol=1e-3,
        lambda_=1,
        alpha=1,
        max_iter=1000,
        plot_sol=True,
        plot_conv=True,
    )

    # Run optimizer
    optimizer_2.minimize(t, y_t, x0)

    # TEST CASE 2...

    print("Running Test case 2")
    print("Function is f(x) = ...")


if __name__ == "__driver_code__":
    main()
