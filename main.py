#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:00:19 2022

@author: mattias
"""

"""Implementation of the Lewenberg Marquardt algorithm, optimization method for nonlinear least squares"""

import numpy as np
import scipy
from scipy.misc import derivative
import pandas as pd
import math
from sympy import symbols, diff

def levmarq(func, x0, t, y_t):
    
    #x1, x2 = symbols('x1 x2', real=True)
    #F = func
    x1 = x0[0]
    x2 = x0[1]
    
    damp = lambda_*np.eye(2)

    F = lambda t, x1, x2: y_t - x1*np.exp(x2*t)
    
    #F_x = np.zeros(t.size)
    F_x = [F(i, x1, x2) for i in t]
    F_x_arr = np.array(F_x)
    
   # F = y_t - x1*np.exp(x2*t)
    #grad_F = lambda t: derivative(F, x0, dx=1e-10)¨
    n = t.size
    grad_F = lambda t, x1, x2: np.array([-np.exp(x2*t), -t*x1*np.exp(x2*t)])
    grad_Fx_arr = np.array([grad_F(i, x1, x2) for i in t])
    
   # grad_F_x1 = diff(F, x1)
  #  grad_F_x2 = diff(F, x2)
    
    while n_iterations <= max_iter and minima == False:
        
        F_x_arr = np.array([F(i, x1, x2) for i in t])
        grad_Fx_arr = np.array([grad_F(i, x1, x2) for i in t])
        
        Xk_1 = Xk - alpha*(grad_Fx_arr @ F_x_arr)).T @ (np.inv(F(t)* grad_F(t) + damp))
        
        if 
        Xk = Xk_1


    return NotImplemented


F = lambda t, x1, x2, y_t: y_t - x1*np.exp(x2*t)

t = np.arange(0.5, 4.5, 0.5)
y = np.array([7.2, 3.0, 1.5, 0.85, 0.48, 0.25, 0.20, 0.15])
#levmarq(F, x0, t, y)

F = lambda t, x1, x2, y_t: y_t - x1*np.exp(x2*t)

grad_F = lambda t: derivative(F, x0, dx=1e-10)

grad_F()