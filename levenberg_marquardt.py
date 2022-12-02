#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:59:23 2022

@authors: Mattias Holmström, Jakob Nyström
"""

import numpy as np
import matplotlib.pyplot as plt


class LevenbergMarquardt:
    """
    Finds the least-squares solution for a nonlinear function using the
    Levenberg-Marquart (LM) algorithm.
    """

    def __init__(
            self,
            func,
            grad=None,
            tol=1e-3,
            lambda_=1,
            alpha=1,
            max_iter=1000,
            plot_conv=False,
    ):
        """
        Initializes the LevenbergMarquardt optimization class object.

        Args:
            func: the function to which we are fitting the least squares solution
            grad: gradient of the function, if provided by the user
            tol: how close to zero that the gradient vector has to be for convergence
            lambda_: dampening factor used in the LM to replace the omitted term in the
                Gauss-Newton formula
            alpha: step size when updating x_k to x_k + 1
            max_iter: max number of iterations allowed before the algorithm terminates
            plot_conv: set to True if MSE and parameter values per iteration should
                be plotted
        """

        # Store input parameters as class attributes
        self.func = func
        self.gradient = grad
        self.tol = tol
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_iter = max_iter
        self.plot_conv = plot_conv

        # Variables for storing results in optimization
        self.function_values = []
        self.param_values = []
        self.final_gradient = None
        self.final_diff = None
        self.final_fx = None
        self.final_x_k = None

        # Attribute for tracking succesful convergence
        self.success = False
        self.n_iterations = 0

    def calculate_function(self, x_k):
        """
        Calculate the mean squared error (MSE) for the given data points and the
        function value for the current parameter values, i.e. the mean of (y - f(x))^2
        for all datapoints.

        Args:
            x_k: the current set of parameter values

        Returns:
            fx: vector of function values at each data point, for current parameters
            mse: mean squared error for current parameters
        """

        # Call supplied function to calculate f(x) for each data point
        fx = self.func(x_k)
        fx = fx.reshape(-1, )

        # Calculate MSE
        mse = (fx @ fx.T) / len(fx)

        return fx, mse

    def calculate_gradient(self, x_k):
        """
        Calculates the gradient for the current parameter values. This is either done
        using a gradient function supplied as an argument by the user, or numerically
        given a numpy array of function values.

        Args:
            x_k: the current set of parameter values

        Returns:
            grad_fx: the gradient matrix for the current point
        """

        if self.gradient:  # Call function supplied by user if it exists
            grad_fx = self.gradient(x_k)

        else:  # Otherwise, use numerical gradient approximation
            fx, _ = self.calculate_function(x_k)
            num_grad = np.zeros((x_k.shape[0], len(fx)))  # Size of matrix
            h = 1e-3  # Step size for gradient

            # Approximate gradient using central differences
            for i in range(x_k.shape[0]):
                x_k1 = np.zeros(x_k.shape[0])
                x_k1[i] = h

                x_k2 = -x_k1
                x_k1 = x_k1 + x_k
                x_k2 = x_k2 + x_k

                fx_1, _ = self.calculate_function(x_k1)
                fx_2, _ = self.calculate_function(x_k2)

                num_grad[i, :] = (fx_1 - fx_2) / (2 * h)

            grad_fx = num_grad

        return grad_fx

    def update_lambda(self, x_k):
        """
        Update value of dampening factor lambda depending on if the MSE increases
        or decreases.
        """

        if self.function_values[self.n_iterations] >= \
                self.function_values[self.n_iterations - 1]:
            damp = self.lambda_ * 1.5 * np.eye(x_k.shape[0])
        else:
            damp = self.lambda_ / 5 * np.eye(x_k.shape[0])

        return damp

    def run_lm_algorithm(self, x0):
        """
        Finds the least-squares solution for a nonlinear function using the
        Levenberg-Marquart (LM) algorithm.

        Args:
            x0: starting guess for parameter values
        """

        # Initial values for parameters
        x_k = np.array(x0)

        # Dampening factor
        damp = self.lambda_ * np.eye(x_k.shape[0])

        # Initialize flag and count variables
        self.n_iterations = 0

        # Loop until optimum or other stopping criteria reached
        while self.n_iterations < self.max_iter:

            # Save parameter values for this iteration
            self.param_values.append(x_k)

            # Compute F(x) vectors and gradient matrix of F(x)
            fx, mse = self.calculate_function(x_k)
            self.function_values.append(mse)
            grad_fx = self.calculate_gradient(x_k)

            # Compute next point x_k+1
            x_k = x_k - self.alpha * (np.linalg.inv(grad_fx @ grad_fx.T + damp)) @ \
                  (grad_fx @ fx)

            # Update lambda matrix depending on increase / decrease in function value
            if self.n_iterations >= 1:
                damp = self.update_lambda(x_k)

            # If gradient vector sufficiently close to zero, break while loop
            if np.sum(np.abs(grad_fx @ fx)) <= self.tol:
                self.success = True
                break

            self.n_iterations += 1

        # Calculate final vectors and values at termination, for output
        self.final_gradient = grad_fx @ fx
        self.final_x_k = x_k
        self.final_fx = fx

    def minimize(self, x_k):
        self.run_lm_algorithm(x_k)
        self.print_output_report()

        if self.plot_conv:
            self.plot_convergence()

        return self.final_x_k, self.function_values[-1], self.final_fx

    def print_output_report(self):
        """Prints results of the optimization in digestable format"""

        # Print results
        print("------ Output report ------\n")
        print(f"Successful convergence: {self.success}")
        print(f"Parameter values: {self.final_x_k}")
        print(f"Function value (MSE): {self.function_values[-1]}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Final gradient vector: {self.final_gradient}")
        print(f"Residual vector: {self.final_fx}")

    def plot_convergence(self):
        """
        Plot function value and parameter values for each iteration, to analyze
        convergence behavior of the algorithm.
        """

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plt.suptitle("Convergence behavior", fontsize=14)

        # Plot MSE
        axes[0].plot(range(len(self.function_values)), self.function_values, color="blue")
        axes[0].set_title("Mean squared error (function value)", fontsize=12)
        axes[0].set_xlabel("iteration", fontsize=12)
        axes[0].set_ylabel("Mean squared error (MSE)", fontsize=12)

        # Plot parameter values
        if len(self.param_values[0]) > 1:
            x_1 = [x_k[0] for x_k in self.param_values]
            x_2 = [x_k[1] for x_k in self.param_values]
            axes[1].scatter(x_1, x_2)
            axes[1].set_xlabel("x_1", fontsize=12)
            axes[1].set_ylabel("x_2", fontsize=12)
        else:
            x_1 = [x_k for x_k in self.param_values]
            axes[1].scatter(x_1)
            axes[1].set_xlabel("x_1", fontsize=12)
            axes[1].set_ylabel("", fontsize=12)

        axes[1].set_title("Parameter values (max two shown)", fontsize=12)

        plt.show()
