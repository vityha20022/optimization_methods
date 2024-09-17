import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def line_search(f, x, d):
    alpha = 0.01
    beta = 0.5
    t = 1.0
    f_x = f(x)
    while f(x + t * d) > f_x + alpha * t * np.dot(d, d):
        t *= beta
    return t

def powell_method(f, x0, num_iter=100, tol=1e-6):
    n = len(x0)
    directions = np.eye(n)
    x = x0

    for _ in range(num_iter):
        f_values = np.array([f(x + d) for d in directions])
        x_best_index = np.argmin(f_values)
        x_best = x + directions[x_best_index]
        
        for i in range(n):
            t = line_search(f, x, directions[i])
            x += t * directions[i]
        
        x_new = x + (x_best - x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, f(x)

initial_point = [1.0, 1.0]
minimum, minimum_value = powell_method(f, initial_point)
print(f"Найденный минимум: {minimum}, значение функции в минимуме: {minimum_value}")
