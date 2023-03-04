import numpy as np
import matplotlib.pyplot as plt

# Definition of functions

# Quadratic function


def basic_function(X): return X[0]**2 + X[1]**2


def grad_basic_function(X): return np.array([2*X[0], 2*X[1]])


def function_for_plotting(X1, X2):
    "FOR PLOTTING Returns the value of function"
    return (np.array(X1)**2 + np.array(X2)**2)

# Rosenbrock function


def compute_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 1.):
    "Returns the value of the banana of rosenbrock given a x, y and a and b parameters"
    x, y = X
    return (a - x)**2 + b * (y - x ** 2)**2


def compute_grad_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 1.):
    "Returns the value of the gradient of the banana of rosenbrock given a x,y and 2 parameters a and b"
    x, y = X
    return np.array([2 * (x-a) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


# Functions for printing


def gradient_descent_print_header():
    print("{:^7}".format("Iter.") + " | " + "{:^10}".format("f(x)") +
          " | " + "{:^10}".format("||df(x)||") + " | " + "{:^10}".format("||d||"))


def gradient_descent_print_iteration(iter, f_iter, norm_g_iter, alpha):

    print(("%7d" % iter) + " | " + ("%.4e" % f_iter) + " | " +
          ("%.4e" % norm_g_iter) + " | " + ("%.4e" % alpha))

# Gradient descent


def grad_descent(f, grad_f, X_init: np.ndarray, eps: float, MAX_ITER: int):
    """ Gradient descent of the f function.

    Inputs : 

    f (function): Function to minimize.
    g (function): Gradient of the function.
    X_init (np.ndarray): Initial start of the function
    eps (float): Tolerance.
    MAX_ITER (int): Number of iteration maximum before not converging.

    Output : 

    X_min (np.ndarray): Coordinates of the minimum of the function
    f_min (np.ndarray): Value of the minimum of the function 
    """

    f_iter = f(X_init)  # Calculating the initial cost of the function
    g_iter = grad_f(X_init)  # Calculating the initial value of the gradient
    iter = 0  # Iteration of the problem
    alpha = 1  # Alpha used for the steps
    X_iter = X_init
    list_f = []
    while np.linalg.norm(g_iter) > eps and iter < MAX_ITER:

        # Printing the iteration and its characteristics
        gradient_descent_print_iteration(
            iter, f_iter, np.linalg.norm(g_iter), alpha)

        # Computing the approximative optimum of the alpha by using a linesearch
        # alpha = backtracking_line_search(f, grad_f, - grad_f(X_iter), X_iter)
        alpha = backtracking_linesearch(f, grad_f, f_iter, g_iter, X_iter)

        # Computing the new X_iter and the new value of the cost function & its gradient
        X_iter = X_iter - alpha * grad_f(X_iter)
        f_iter = f(X_iter)
        g_iter = grad_f(X_iter)
        iter += 1

        list_f.append(f_iter)
    return X_iter, f_iter, list_f


# Linesearch


def backtracking_linesearch(f, g, f_iter, grad_f_iter, X_iter):
    """Backtracking line search using Armijo-Goldstein condition.

    Arguments:
        f -- function to minimize
        g -- gradient of the function
        f_iter -- value of the function at the iter-th step
        grad_f_iter -- value of the gradient of the function at the iter-th step
        X_iter -- value of X at the iter-th step

    Returns:
        alpha -- Approximative value of the alpha
    """
    max_it = 20
    it = 0
    alpha = 1
    alpha_decrease = 0.1
    phi = 1e-2
    # p_k in Nocedal, direction of the linesearch (- gradient for the gradient descent)
    p_iter = - grad_f_iter
    while it < max_it:
        if f_iter - f(X_iter + alpha * p_iter) >= - alpha * phi * grad_f_iter.T @ p_iter:
            return alpha
        alpha *= alpha_decrease
        it += 1
    return alpha


if __name__ == "__main__":

    X_init = np.array([2, 3])  # Initial start of the function
    eps = 1e-6  # Tolerance
    MAX_ITER = 10000  # Number max of iteration
    gradient_descent_print_header
    X, f_iter, list_f = grad_descent(
        basic_function, grad_basic_function, X_init, eps, MAX_ITER)

    # Plotting the results
    x1 = np.linspace(-10, 10)
    x2 = np.linspace(-10, 10)
    X1, X2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X1, X2, function_for_plotting(X1, X2))
    plt.xlabel("X value")
    plt.ylabel("Y value")
    ax1.set_zlabel("F(X,Y)")

    ax2 = fig.add_subplot(122)
    ax2.plot(list_f)
    plt.xlabel("Number of iterations")
    plt.ylabel("Value of the function")
    plt.title(" Basic function")
    plt.show()
