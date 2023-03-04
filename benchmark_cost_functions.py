import numpy as np

def square_sum(X: np.ndarray): return np.sum(np.square(X))

def grad_square_sum(X: np.ndarray): return 2 * np.sum(X)

def log_sum(X: np.ndarray): return np.log(X)

