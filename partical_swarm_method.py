import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape)
        self.best_position = self.position.copy()
        self.best_value = f(self.position)

def particle_swarm_optimization(f, bounds, num_particles, num_iterations):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = min(particles, key=lambda p: p.best_value).best_position
    global_best_value = f(global_best_position)

    for _ in range(num_iterations):
        for particle in particles:
            r1, r2 = np.random.rand(2)
            particle.velocity = (0.5 * particle.velocity +
                                 r1 * (particle.best_position - particle.position) +
                                 r2 * (global_best_position - particle.position))
            particle.position += particle.velocity
            
            if np.all(bounds[:, 0] <= particle.position) and np.all(particle.position <= bounds[:, 1]):
                value = f(particle.position)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = particle.position.copy()
                    
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particle.position.copy()

    return global_best_position, global_best_value

bounds = np.array([[-5, 5], [-5, 5]])
num_particles = 30
num_iterations = 100

minimum, minimum_value = particle_swarm_optimization(f, bounds, num_particles, num_iterations)
print(f"Найденный минимум: {minimum}, значение функции в минимуме: {minimum_value}")
