import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def box_complex_method(f, x0, step_size=1.0, tol=1e-6, max_iter=100):
    n = len(x0)
    simplex = np.array([x0 + step_size * np.eye(n)[i] for i in range(n)])
    simplex = np.vstack([x0, simplex])
    
    for iteration in range(max_iter):
        f_values = np.array([f(s) for s in simplex])
        sorted_indices = np.argsort(f_values)
        simplex = simplex[sorted_indices]
        f_values = f_values[sorted_indices]
        
        centroid = np.mean(simplex[:-1], axis=0)
        reflection = centroid + (centroid - simplex[-1])
        f_reflection = f(reflection)

        if f_values[0] <= f_reflection < f_values[-2]:
            simplex[-1] = reflection
        elif f_reflection < f_values[0]:
            expansion = centroid + 2 * (centroid - simplex[-1])
            f_expansion = f(expansion)
            if f_expansion < f_reflection:
                simplex[-1] = expansion
            else:
                simplex[-1] = reflection
        else:
            contraction = centroid + 0.5 * (simplex[-1] - centroid)
            f_contraction = f(contraction)
            if f_contraction < f_values[-1]:
                simplex[-1] = contraction
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])

        if np.max(np.abs(f_values - f_values[0])) < tol:
            break

    return simplex[0], f_values[0]

initial_point = [1.0, 1.0]
minimum, minimum_value = box_complex_method(f, initial_point)
print(f"Найденный минимум: {minimum}, значение функции в минимуме: {minimum_value}")
