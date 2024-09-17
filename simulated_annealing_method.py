import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def simulated_annealing(f, initial_solution, initial_temperature, cooling_rate, num_iterations):
    current_solution = initial_solution
    best_solution = current_solution
    best_value = f(best_solution)
    temperature = initial_temperature

    for iteration in range(num_iterations):
        new_solution = current_solution + np.random.uniform(-1, 1, size=current_solution.shape)
        new_value = f(new_solution)
        acceptance_probability = np.exp((f(current_solution) - new_value) / temperature)

        if new_value < best_value or np.random.rand() < acceptance_probability:
            current_solution = new_solution
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value
        
        temperature *= cooling_rate

    return best_solution, best_value

initial_solution = np.array([5.0, 5.0])
initial_temperature = 1000
cooling_rate = 0.99
num_iterations = 1000

minimum, minimum_value = simulated_annealing(f, initial_solution, initial_temperature, cooling_rate, num_iterations)
print(f"Найденный минимум: {minimum}, значение функции в минимуме: {minimum_value}")
