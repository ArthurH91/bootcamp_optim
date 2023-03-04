import numpy as np
import matplotlib.pyplot as plt


# Definition of the rosenbrock function

def compute_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 100.):
    "Returns the value of the banana of rosenbrock given a x, y and a and b parameters"
    x, y = X
    return (a - x)**2 + b * (y - x ** 2)**2


def compute_rosenbrock_function1(X1,X2, a: float = 1., b: float = 100.):
    "FOR PLOTTING Returns the value of the banana of rosenbrock given a x, y and a and b parameters"
    return (a - np.array(X1))**2 + b * (np.array(X2) - np.array(X1) ** 2)**2


def compute_grad_rosenbrock_function(X: np.ndarray, a: float = 1., b: float = 100.):
    "Returns the value of the gradient of the banana of rosenbrock given a x,y and 2 parameters a and b"
    x, y = X
    return np.array([2 * (x-a) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])


def compute_hess_ros_function(X: np.ndarray, a: float = 1, b: float = 100.):
    "Returns the hessian value of the rosenbrock function given a x, y and 2 parameters a and b"
    x, y = X
    H = np.eye(2)
    H[0, 0] = 2 - 4 * b * (y - 3 * x ** 2)
    H[0, 1] = - 4 * b * x
    H[1, 0] = - 4 * b * x
    H[1, 1] = 2 * b
    return H


# Functions to find the minimum with a linesearch

def compute_newton_direction(X: np.ndarray, a: float = 1., b: float = 100):
    "Returns the newton direction, which is H^-1 * grad(f)"
    x, y = X
    Hessian = compute_hess_ros_function(x, y, a, b)
    Gradient = compute_grad_rosenbrock_function(x, y, a, b)
    return - np.linalg.pinv(Hessian) * Gradient


def newton(X_init: np.ndarray, eps: float = 1e-5, max_it:int = 1000, a: float = 1., b: float = 100.):
    "Returns the min of the rosenbrock function along the number of iteration and the list of steps with the Newton Method"
    X = X_init
    it = 0
    X_list = [X_init]
    while it < max_it:
        J_grad_X, J_hess_X = compute_grad_rosenbrock_function(X,a,b), compute_hess_ros_function(X,a,b)
        g = np.linalg.solve(J_hess_X, J_grad_X)
        X = X - g 
        X_list.append(X)
        if np.linalg.norm(J_grad_X) < eps:
            return X, it +1, X_list
        it += 1 
    return X, max_it, X_list



def gradient_descent(X_init: np.ndarray, eps: float = 1e-3, max_it = 100000, a: float = 1., b: float = 100.):
    "Returns the min of the rosenbrock function along the number of iterations and the list of steps with the gradient descent method"
    X = X_init
    it = 0
    X_list = [X_init] 
    while it < max_it:
        dt = 1e-3
        J_grad_X = compute_grad_rosenbrock_function(X,a,b)
        gradient_dir = - dt * J_grad_X
        X = X + gradient_dir
        X_list.append(X)
        if np.linalg.norm(J_grad_X)<eps:
            return X, it + 1, X_list
        it = it + 1
    return X, max_it, X_list

def gradient_descent_linesearch(X_init: np.ndarray, eps: float = 1e-3, phi: float = 1e-5, max_it = 100000, a: float = 1., b: float = 100.):
    "Returns the min of the rosebrock function along the number of iterations and the list of steps with the gradient descent method with a linesearch "
    X = X_init
    it = 0
    X_list = [X_init]
    while it < max_it:
        J_grad_X = compute_grad_rosenbrock_function(X, a, b)
        dt = linesearch(X, J_grad_X, phi, a, b)
        gradient_dir = - dt * phi * J_grad_X
        X = X + gradient_dir
        X_list.append(X)
        if np.linalg.norm(J_grad_X) < eps:
            return X, it + 1, X_list
        it = it + 1
    return X, max_it, X_list

def linesearch(X: np.ndarray, J_grad_X: np.ndarray, phi: float = 1e-5, a: float = 1., b: float = 100.):
    "Returns an approximation of the dt optimum used in the gradient descent"
    max_it = 20
    it = 0
    dt = 1e8
    dt_decrease = 0.1
    while it < max_it:
        gradient_dir = - phi * dt * J_grad_X
        X_new = X + gradient_dir
        J_grad_X_new = compute_grad_rosenbrock_function(X_new,a,b)
        if np.linalg.norm(J_grad_X_new) < np.linalg.norm(J_grad_X): # Armijio condition
            return dt
        dt *= dt_decrease
        it += 1
    return dt
    

if __name__ == "__main__":

    # Declaration of the parameters a and b :
    # f (x,y) = (a - x)**2 + b * (y - x ** 2)**2

    a = 1.
    b = 100.

    # Declaration of the working range

    eps = 1e-3 # for the global algorithm
    X_init = np.zeros(2)
    max_it = 100000
    phi = 1e-5 # for the linesearch

    
    ## NEWTON METHOD 
    

    X, it, X_list = newton(X_init,eps,max_it)
    print("Newton method :")    
    if it == max_it:
        print("Did not converge successfully")
    else:
        print("Converged sucessfully !")
    print(f"Number of iterations : {it}")
    print(f"Value of X : {X}")    
    print("-----------------------")
    
    
    X_newton = []
    Y_newton = []
    Rosenbrock_result_Newton = []
    for k in range(len(X_list)):
        Xx = X_list[k][0]
        Xy = X_list[k][1]
        X_newton.append(Xx)
        Y_newton.append(Xy)
        Rosenbrock_result_Newton.append(compute_rosenbrock_function1(Xx,Xy,a,b))
    
    
    
    step = np.arange(0,len(Rosenbrock_result_Newton))
    plt.subplot(221)
    plt.plot(X_newton, Y_newton)
    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.title("Newton method")

    
    
    ## GRADIENT DESCENT METHOD
    
    X, it, X_list = gradient_descent(X_init, eps, max_it)
    print("Gradient descent method :")
    if it == max_it:
        print("Did not converge successfully")
    else:
        print("Converged sucessfully !")
    print(f"Number of iterations : {it}")
    print(f"Value of X : {X}")
    print("-----------------------")


    X_gd = []
    Y_gd = []
    for k in range(len(X_list)):
        X_gd.append(X_list[k][0])
        Y_gd.append(X_list[k][1])

    plt.subplot(222)
    plt.plot(X_gd, Y_gd)
    plt.title("Gradient descent method")    
    plt.xlabel("X value")
    plt.ylabel("Y value")


    ## GRADIENT DESCENT METHOD WITH ALPHA FOUND WITH A LINESEARCH
    
    X, it, X_list = gradient_descent_linesearch(X_init, eps, phi, max_it)
    print("Gradient descent method with a linesearch :")
    if it == max_it:
        print("Did not converge successfully")
    else:
        print("Converged sucessfully")
    print(f"Number of iterations : {it}")
    print(f"Value of X : {X}")
    print("-----------------------")

    X_gd_l = []
    Y_gd_l = []
    for k in range(len(X_list)):
        X_gd_l.append(X_list[k][0])
        Y_gd_l.append(X_list[k][1])

    plt.subplot(223)
    plt.plot(X_gd_l, Y_gd_l)
    plt.title("Gradient descent method with a linesearch")
    plt.xlabel("X value")
    plt.ylabel("Y value")
    plt.show()
    



    
    # Plotting the function
    
    x1 = np.linspace(-10, 10)
    x2 = np.linspace(-10, 10)
    X1, X2 = np.meshgrid(x1, x2)
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X1, X2,compute_rosenbrock_function1(X1,X2,a,b))
    plt.xlabel("X value")
    plt.ylabel("Y value")
    ax1.set_zlabel("F(X,Y)")
    
    plt.title("Rosenbrock function")    

    
    
    # Plotting the results in 3D
    
    
    
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot(X_newton,Y_newton,compute_rosenbrock_function1(X_newton,Y_newton,a,b), "o")
    plt.title("Newton method")    
    plt.xlabel("X value")
    plt.ylabel("Y value")
    ax2.set_zlabel("F(X,Y)")
    
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot(X_gd,Y_gd,compute_rosenbrock_function1(X_gd,Y_gd,a,b), "o")
    plt.title("Gradient descent method")
    plt.xlabel("X value")
    plt.ylabel("Y value")
    ax3.set_zlabel("F(X,Y)")
    
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot(X_gd_l, Y_gd_l, compute_rosenbrock_function1(X_gd_l, Y_gd_l, a, b), "o")
    plt.title("Gradient descent method with a linesearch")
    plt.xlabel("X value")
    plt.ylabel("Y value")
    ax4.set_zlabel("F(X,Y)")
    
    
    
    
    plt.show()
    
    