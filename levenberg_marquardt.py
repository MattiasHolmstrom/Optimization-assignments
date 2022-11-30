#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 14:59:23 2022

@author: Mattias Holmström, Jakob Nyström
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff



class LevenbergMarquardt:
    """
    Finds the least-squares solution for a nonlinear function using the
    Levenberg-Marquart (LM) algorithm.
    """

    def __init__(
            self,
            func,
            grad=None,
            x_vec=None,
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
        
       # params = [symbols(f'{symbol}') for symbol in x_vec]
        
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
        Calculate the mean squared error (MSE) for the y values of the given data
        points and the function value for the current parameter values.
        """

        # Call supplied function to calculate f(x) for each data point
        fx = self.func(x_k)

        fx = fx.reshape(-1, )

        # Calculate MSE: (1 / n) * (y - y_hat)^2
        mse = (fx @ fx.T) / len(fx)

        return fx, mse

    def calculate_gradient(self, fx, x_k):
        """
        Calculates the gradient numerically given a numpy array of function values.
        """

        if self.gradient:  # Call function supplied by user if it exists
            grad_fx = self.gradient(x_k)
        else:  # Otherwise, use numerical gradient approximation
            num_grad = np.zeros((x_k.shape[0], 8))
            h = 1e-1
            
            for i in range(x_k.shape[0]):
                x_k1 = x_k
                x_k2 = x_k

                x_k1[i] = x_k1[i]+h
                x_k2[i] = x_k2[i]-h

                f, e= self.calculate_function(x_k1)
                f2, e2 = self.calculate_function(x_k2)
                
                num_grad[i, :] = (f-f2)/(2*h)
                
            #grad = np.gradient(fx)
            grad_fx = num_grad

        return grad_fx
        
        

    def run_lm_algorithm(self, x0):
        """
        Finds the least-squares solution for a nonlinear function using the
        Levenberg-Marquart (LM) algorithm.

        Args:
            x0: starting guess for parameter values

        Returns:
            x_k: optimal parameter values
            function_values: value of the function for each iteration
            parameter_values value of parameters for each iteration
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
            grad_fx = self.calculate_gradient(fx, x_k)
            

            # Compute next point x_k+1
            x_k = x_k - self.alpha * (np.linalg.inv(grad_fx @ grad_fx.T + damp)) @ \
                  (grad_fx @ fx)
                  
            if self.n_iterations >= 1:
               if self.function_values[self.n_iterations] >= self.function_values[self.n_iterations-1]:
                   damp = self.lambda_*1 * np.eye(x_k.shape[0])
               else:
                   damp = self.lambda_/5 * np.eye(x_k.shape[0])

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
        print(f"Function value: {self.function_values[-1]}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Final gradient vector: {self.final_gradient}")

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
