import numpy as np

def generate_test_function(num_dimensions, function_type='quadratic'):
    if function_type == 'quadratic':
        def f(x):
            return np.sum(x**2)
    elif function_type == 'cubic':
        def f(x):
            return np.sum(x**3)
    elif function_type == 'nonconvex':
        def f(x):
            return np.sin(np.sum(x**2)) + np.sum(x**2)
    else:
        raise ValueError("Unknown function type")

    return f

def generate_test_data(num_samples, num_dimensions, bounds, function_type='quadratic'):
    function = generate_test_function(num_dimensions, function_type)
    data = []

    for _ in range(num_samples):
        x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=num_dimensions)
        value = function(x)
        data.append((x, value))

    return data

num_samples = 100
num_dimensions = 2
bounds = np.array([[-5, 5], [-5, 5]])
test_data = generate_test_data(num_samples, num_dimensions, bounds, function_type='quadratic')

for x, value in test_data:
    print(f"Точка: {x}, Значение: {value}")
