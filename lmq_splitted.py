#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:59:23 2022

@author: Mattias Holmström, Jakob Nyström
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
            func=None,
            grad=None,
            tol=1e-3,
            lambda_=1,
            alpha=1,
            max_iter=1000,
            plot_sol=False,
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
            plot_sol:
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
        self.plot_sol = plot_sol
        self.plot_conv = plot_conv

        # Variables for storing results in optimization
        self.function_values = []
        self.param_values = []
        self.final_gradient = None
        self.final_x_k = None

        # Attribute for tracking succesful convergence
        self.success = False
        self.n_iterations = 0

    @staticmethod
    def calculate_function_val(t, y_t, x_k):
        """
        Calculate the sum of least squares errors for the function supplied to the
        optimization algorithm, for a given set of parameter values (at one point).
        """

        F = lambda i, x_k: y_t[i] - x_k[0] * np.exp(x_k[1] * t[i])
        Fx = np.array([F(i, x_k) for i in range(len(t))])
        function_value = Fx @ Fx.T
        return Fx, function_value

    @staticmethod
    def calculate_gradient(t, x_k):
        """
        Calculate the gradient of the supplied function using finite differencing,
        unless the gradient is provided by the user.
        """

        grad_F = lambda i, x_k: np.array(
            [
                -np.exp(x_k[1] * t[i]),
                -t[i] * x_k[0] * np.exp(x_k[1] * t[i])
            ]
        )
        grad_Fx = np.array([grad_F(i, x_k) for i in range(len(t))]).T
        return grad_Fx

    def run_lm_algorithm(self, t, y_t, x0):
        """
        Finds the least-squares solution for a nonlinear function using the
        Levenberg-Marquart (LM) algorithm.

        Args:
            t: input arguments to the function
            y_t: true values of the function
            x0: starting guess for parameter values

        Returns:
            x_k: optimal parameter values
            function_values: value of the function for each iteration
            parameter_values value of parameters for each iteration
        """

        # Initial values for parameters
        x_k = np.array(x0)

        # Dampening factor
        damp = self.lambda_ * np.eye(2)

        # Initialize flag and count variables
        self.n_iterations = 0

        # Loop until optimum or other stopping criteria reached
        while self.n_iterations < self.max_iter:
            self.n_iterations += 1

            # Save parameter values for this iteration
            self.param_values.append(x_k)

            # Compute F(x) vectors and gradient matrix of F(x)
            Fx, function_value = self.calculate_function_val(t, y_t, x_k)
            grad_Fx = self.calculate_gradient(t, x_k)
            self.function_values.append(function_value)

            # Compute next point x_k+1
            x_k = x_k - self.alpha * (np.linalg.inv(grad_Fx @ grad_Fx.T + damp)) @ \
                  (grad_Fx @ Fx)

            if np.sum(np.abs(grad_Fx.T @ Fx)) <= self.tol:
                self.success = True
                break

            # Calculate the final gradient vector at termination, for output
            self.final_gradient = grad_Fx.T @ Fx
            self.final_x_k = x_k

    def print_output_report(self):
        """Prints results of the optimization in digestable format"""

        # Print results
        print("------ Output report ------\n")
        print(f"Successful convergence: {self.success}")
        print(f"Parameter values: {self.final_x_k}")
        print(f"Function value: {self.function_values[-1]}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Final gradient vector: {self.final_gradient}")

    def minimize(self, t, y_t, x_k):
        self.run_lm_algorithm(t, y_t, x_k)
        self.print_output_report()

        # Generate plots if set to True
        if self.plot_sol:
            self.plot_solution(t, y_t)
        if self.plot_conv:
            self.plot_convergence()

    def plot_solution(self, t, y):
        """
        Plot the solution given by the optimized parameters and the original
        datapoints.
        """

        # Create plot
        plt.figure(figsize=(6, 4))
        plt.title("Least squares solution", fontsize=14, pad=16)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("f(x)", fontsize=12)

        # Calculate least squares solution (line)
        line_range = np.arange(min(t), max(t), 0.1)
        line_values = [self.final_x_k[0] * np.exp(self.final_x_k[1] * line_range[i]) for i in range(len(line_range))]

        # Plot solution and original datapoints
        plt.scatter(t, y, color="red")
        plt.plot(line_range, line_values, color="blue")
        plt.show()

    def plot_convergence(self):
        """
        Plot function value and parameter values for each iteration, to analyze
        convergence behavior of the algorithm.
        """
        # TODO: Add plot for x_k

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        plt.suptitle("Convergence behavior", fontsize=14)

        # Plot MSE and parameter values
        axes[0].plot(range(len(self.function_values)), self.function_values, color="blue")
        axes[0].set_title("Mean squared error (function value)", fontsize=12)
        axes[0].set_xlabel("iteration", fontsize=12)
        axes[0].set_ylabel("Mean squared error (MSE)", fontsize=12)

        # axes[0, 1].plot
        plt.show()
