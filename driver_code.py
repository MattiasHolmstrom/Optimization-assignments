
import numpy as np
from levenberg_marquardt import LevenbergMarquardt


def main():

    # TEST CASE 1 (given in the assignment)

    print("Running Test case 1 (given in assignment)\n")
    print("Function is: y(t) = x_1 * e^(x_2 * t)\n")

    # Define values of t, y_t and x0
    t = np.arange(0.5, 4.5, 0.5)
    y_t = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
    x0 = np.array([1, 1])

    # Define test function
    def test_function_1(x_k):
        fx = y_t - (x_k[0] * np.exp(x_k[1] * t))

        return fx

    # Define gradient function
    def test_gradient_1(x_k):
        grad_fx = np.array([-np.exp(x_k[1] * t), -t * x_k[0] * np.exp(x_k[1] * t)])

        return grad_fx

    # Initialize optimizer WITH gradient argument
    optimizer_1 = LevenbergMarquardt(
        func=test_function_1,
        grad=test_gradient_1,
        tol=1e-3,
        lambda_=1,
        alpha=1,
        max_iter=1000,
        plot_conv=True,
    )

    # Run optimizer
    optimizer_1.minimize(x0)
    optimizer_1.plot_solution(t, y_t)

    # TEST CASE 2...

    print("Running Test case 2")
    print("Function is f(x) = ...")


if __name__ == "__driver_code__":
    main()
