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

class Lmq:
    def __init__(self, func, gradient, x_vec, t, y):
        self.func = func
       # self.gradient = 
        #self.x_vec = 
        
    
    def calculate_function_val(func, y_t, t, x_k):
        
        func_vec = func()
        
        return NotImplementedError()
    def __str__():
        
        return NotImplementedError()
    def calculate_gradient(func, of_x):
        
        grad_f = np.array([integrate(func, x_i) for x_i in of_x])
        print(grad_f)
        return grad_f


    f = lambda y_t, x1, x2: y_t - x1 * np.exp(x2 * t)

    calculate_gradient(f, )

        
def levmarq(t, y_t, x0, tol=1e-3, lambda_=1, alpha=1, max_iter=1000):
    """
    Finds the least-squares solution for a nonlinear function using the
    Levenberg-Marquart (LM) algorithm.

    Args:
        t: input arguments to the function
        y_t: true values of the function
        x0: starting guess for parameter values
        tol: how close to zero that the gradient vector has to be for convergence
        lambda_: dampening factor used in the LM to replace the omitted term in the
            Gauss-Newton formula
        alpha: step size when updating x_k to x_k + 1
        max_iter: maximum number of iterations allowed before the algorithm terminates

    Returns:
        x_k: optimal parameter values
        function_values: value of the function for each iteration
        parameter_values value of parameters for each iteration

    """

    # Initial values for parameters
    x_k = np.array(x0)

    # Dampening factor
    damp = lambda_ * np.eye(2)

    # Lists for storing intermediary results
    function_values = []
    param_values = []
    x_k_values = []

    # Functions to calculate F and gradient of F
    F = lambda i, x_k: y_t[i] - x_k[0] * np.exp(x_k[1] * t[i])
    grad_F = lambda i, x_k: np.array(
        [
            -np.exp(x_k[1] * t[i]),
            -t[i] * x_k[0] * np.exp(x_k[1] * t[i])
        ]
    )

    # Initialize flag and count variables
    n_iterations = 0
    success = False

    # Loop until optimum or other stopping criteria reached
    while n_iterations < max_iter:
        n_iterations += 1

        # Save parameter values for this iteration
        param_values.append(x_k)

        # Compute F(x) vectors and gradient matrix of F(x)
        Fx = np.array([F(i, x_k) for i in range(len(t))])
        grad_Fx = np.array([grad_F(i, x_k) for i in range(len(t))])
        function_value = Fx @ Fx.T
        function_values.append(function_value)
        
        x_k_values.append(x_k)

        # Compute next point x_k+1
        x_k = x_k - alpha * ((grad_Fx.T @ Fx).T @
                             (np.linalg.inv(grad_Fx.T @ grad_Fx + damp)))

        if np.sum(np.abs(grad_Fx.T @ Fx)) <= tol:
            success = True
            break

    # Print results
    print("------ Output report ------\n")
    print(f"Successful convergence: {success}")
    print(f"Parameter values: {x_k}")
    print(f"Function value: {function_values[:10]}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Gradient vector: {grad_Fx.T @ Fx}")
    print(f"Convergence in x: {x_k_values[-10:]}")

    return x_k, function_values, param_values

t = np.arange(0.5, 4.5, 0.5)
y = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])

levmarq(t, y, np.array([10, -5.0]))

f = [np.sqrt()]