
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy import optimize


class SupportVectorClassifier:

    def __init__(self, solver="SLSQP", penalty=1000, max_iter=1000):

        # Store class attributes
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter

        # Placeholder attributes for data
        self.X = None
        self.Y = None
        self.Y_triang = None
        self.W = None
        self.b = None

    def setup_data_arrays(self):

        # Create 30 x 1 weight vector with initial weights
        # Also create 569 x 1 intercept / bias vector with initial value
        self.W = np.random.normal(loc=0, scale=0.1, size=self.X.shape[0])
        self.b = np.ones(self.X.shape[1])
        # b_test = np.ones(self.X.shape[1])

        # Transform class label vector into triangular matrix with labels on diagonal
        self.Y_triang = np.zeros((self.X.shape[1], self.X.shape[1]))
        np.fill_diagonal(self.Y_triang, self.Y)

    def calculate_value_per_obs(self):
        # Predicted value for each x_i with current weights, times actual class label
        pred_actual_val = (self.W.T @ self.X + self.b) @ self.Y_triang

        return pred_actual_val

    def calculate_epsilon(self):
        # Predicted value for each x_i with current weights, times actual class label
        pred_actual_val = self.calculate_value_per_obs()

        # Epsilon value for each x_i on the wrong side of support vector
        epsilon_vec = np.where(pred_actual_val < 1, 1 - pred_actual_val, 0)

        return epsilon_vec

    def objective_function(self):

        # Call function to calculate value per observation
        pred_actual_val = self.calculate_value_per_obs()

        # Call function to calculate current epsilon vector
        epsilon_vec = self.calculate_epsilon()

        # Calculate value of function for current weights and deviations
        function_val = 0.5 * (self.W.T @ self.W) + self.C * np.sum(epsilon_vec)

        return function_val

    def constraint_function(self):

        # Call function to calculate value per observation
        pred_actual_val = self.calculate_value_per_obs()

        # Call function to calculate current epsilon vector
        epsilon_vec = self.calculate_epsilon()

        # Difference between these
        constraint = pred_actual_val - 1 + epsilon_vec

        return constraint

    def fit_model(self, x_train, y_train):

        # Store data passed to the function
        self.X = x_train
        self.Y = y_train

        # Set up the required arrays
        self.setup_data_arrays()

        # Run optimization
        epsilon_cons = {"type": "ineq", "fun": self.constraint_function}

        optim = optimize.minimize(
            fun=self.objective_function,
            x0=self.W,
            method=self.solver,
            constraints=epsilon_cons,
            options={"maxiter": self.max_iter},
        )

    def predict(self, x_test):

        values = self.W @ x_test
        predictions = np.where(values >= 0, 1, -1)

        return predictions

