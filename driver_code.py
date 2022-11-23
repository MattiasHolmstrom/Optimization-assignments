
import numpy as np
from lmq_splitted import LevenbergMarquardt


def main():

    optimizer = LevenbergMarquardt(
        func=None,
        grad=None,
        tol=1e-3,
        lambda_=1,
        alpha=1,
        max_iter=1000,
        plot_sol=True,
        plot_conv=True,
    )

    t = np.arange(0.5, 4.5, 0.5)
    y = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
    x0 = [1, 1]

    optimizer.minimize_least_squares(t, y, x0)


if __name__ == "__driver_code__":
    main()
