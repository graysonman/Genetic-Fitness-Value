import numpy as np
import random
import math

def f(x, y):
    return (1 - x)**2 * math.exp(-x**2 - (y + 1)**2) - (x - x**3 - y**3) * math.exp(-x**2 - y**2)

def generate_population(size, x_bounds, y_bounds):
    return [(random.uniform(*x_bounds), random.uniform(*y_bounds)) for _ in range(size)]

def crossover(parent1, parent2, crossover_probability):
    if random.random() < crossover_probability:
        point = random.randrange(1, len(parent1))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_probability, x_bounds, y_bounds):
    mutated_individual = []
    for gene, bounds in zip(individual, (x_bounds, y_bounds)):
        if random.random() < mutation_probability:
            mutated_individual.append(random.uniform(*bounds))
        else:
            mutated_individual.append(gene)
    return tuple(mutated_individual)

def selection(population, fitnesses):
    sorted_pop = sorted(population, key=lambda ind: fitnesses[ind], reverse=True)
    return sorted_pop[:len(population) // 2]

def genetic_algorithm(population_size, x_bounds, y_bounds, crossover_probability, mutation_probability, generations):
    population = generate_population(population_size, x_bounds, y_bounds)
    for _ in range(generations):
        fitnesses = {individual: f(*individual) for individual in population}
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(selection(population, fitnesses), 2)
            child1, child2 = crossover(parent1, parent2, crossover_probability)

            child1 = mutate(child1, mutation_probability, x_bounds, y_bounds)
            child2 = mutate(child2, mutation_probability, x_bounds, y_bounds)

            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=lambda ind: fitnesses[ind])
    return best_individual, fitnesses[best_individual]

population_size = 8
x_bounds = (-2, 2)
y_bounds = (-2, 2)
crossover_probability = 0.7
mutation_probability = 0.01
generations = 200

best_xy, best_fitness = genetic_algorithm(population_size, x_bounds, y_bounds, crossover_probability, mutation_probability, generations)

print(f"Best x, y: {best_xy}")
print(f"Best fitness value: {best_fitness}")
