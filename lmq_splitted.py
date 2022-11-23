#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:59:23 2022

@author: Mattias Holmström, Jakob Nyström
"""


import numpy as np
import scipy
from scipy.misc import derivative
import pandas as pd
import math
from sympy import symbols, diff
import matplotlib.pyplot as plt
from sympy import integrate, Symbol


class LevenbergMarquardt:

    """
    Finds the least-squares solution for a nonlinear function using the
    Levenberg-Marquart (LM) algorithm.
    """

    def __init__(self, func, grad=None, tol=1e-3, lambda_=1, alpha=1, max_iter=1000):
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
        """

        # Store input parameters as class attributes
        self.func = func
        self.gradient = grad
        self.tol = tol
        self.lambda_ = lambda_
        self.alpha = alpha
        self.max_iter = max_iter

        # Variables for storing results in optimization
        self.x_k = None
        self.function_values = []
        self.param_values = []
        self.final_gradient = None

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
        grad_Fx = np.array([grad_F(i, x_k) for i in range(len(t))])
        return grad_Fx

    def minimize_least_squares(self, t, y_t, x0):
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
        self.x_k = np.array(x0)

        # Dampening factor
        damp = self.lambda_ * np.eye(2)

        # Initialize flag and count variables
        self.n_iterations = 0

        # Loop until optimum or other stopping criteria reached
        while self.n_iterations < self.max_iter:
            self.n_iterations += 1

            # Save parameter values for this iteration
            self.param_values.append(self.x_k)

            # Compute F(x) vectors and gradient matrix of F(x)
            Fx, function_value = self.calculate_function_val(t, y_t, self.x_k)
            grad_Fx = self.calculate_gradient(t, self.x_k)
            self.function_values.append(function_value)

            # Compute next point x_k+1
            self.x_k = self.x_k - self.alpha * ((grad_Fx.T @ Fx).T @
                                                (np.linalg.inv(grad_Fx.T @ grad_Fx + damp)))

            if np.sum(np.abs(grad_Fx.T @ Fx)) <= self.tol:
                self.success = True
                break

            # Calculate the final gradient vector at termination
            self.final_gradient = grad_Fx.T @ Fx

    def print_output_report(self):

        # Print results
        print("------ Output report ------\n")
        print(f"Successful convergence: {self.success}")
        print(f"Parameter values: {self.x_k}")
        print(f"Function value: {self.function_values[:10]}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Gradient vector: {self.final_gradient}")

    def plot_convergence(self):
        pass
