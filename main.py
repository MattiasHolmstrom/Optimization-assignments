#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:00:19 2022

@author: Mattias Holmström, Jakob Nyström
"""

"""Implementation of the Lewenberg Marquardt algorithm, optimization method for nonlinear least squares"""

import numpy as np
import scipy
from scipy.misc import derivative
import pandas as pd
import math
from sympy import symbols, diff

def levmarq(t, y_t, x0, tol=0.01, lambda_=1, alpha=1, max_iter=100):
    
    # Initial values for parameters
    x_k = np.array(x0)
    
    # Dampening factor
    damp = lambda_ * np.eye(2)
    
    # Lists for storing intermediary results
    function_values = []
    param_values = []

    # Functions to calculate F and gradient of F
    F = lambda t, x_k: y_t - x_k[0] * np.exp(x_k[1] * t)
    grad_F = lambda t, x_k: np.array([-np.exp(x_k[1] * t), -t * x_k[0] * np.exp(x_k[1] * t)])
    
    # Loop until optimum or other stopping criteria reached
    while n_iterations < max_iter:
        
        # Save parameter values for this iteration
        param_values.append(x_k)
        
        # Compute F(x) vectors and gradient matrix of F(x)
        Fx = np.array([F(i, x_k) for i in t])
        grad_Fx = np.array([grad_F(i, x_k) for i in t])
        funcion_value = np.sum(Fx)
        function_values.append(funcion_value)
        
        # Compute next point x_k+1 
        X_k = X_k - alpha * ((grad_Fx @ Fx).T @ (np.inv(grad_Fx @ grad_Fx.T + damp)))
        
        if abs(grad_Fx @ Fx) < tol:
            return (x_1, x_2)

    return (x_1, x_2)

t = np.arange(0.5, 4.5, 0.5)
y = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
