import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def random_search(f, bounds, num_samples=1000):
    best_x = None
    best_value = float('inf')
    
    for _ in range(num_samples):
        x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        value = f(x)
        if value < best_value:
            best_value = value
            best_x = x
            
    return best_x, best_value

bounds = np.array([[-5, 5], [-5, 5]])
minimum, minimum_value = random_search(f, bounds)
print(f"Найденный минимум: {minimum}, значение функции в минимуме: {minimum_value}")
