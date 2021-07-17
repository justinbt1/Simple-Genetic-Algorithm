import numpy as np
import random
from copy import copy


def generate_initial_population(population_n, chromosome_length):
    """ Population_n: Randomly generates as initial population.

    Args:
        population_n(int): Number of individuals in population.
        chromosome_length(int): Number of genes in a chromosome.

    Returns:
        np.array: Initial population.

    """
    population = np.random.randint(2, size=(population_n, chromosome_length))
    return population


def tournament_selection(parent_population, population_size, tournament_size):
    """ Use deterministic tournament selection to select parents.

    Let the Hunger Games commence!

    Args:
        parent_population(np.array): Parent population matrix.
        population_size(int): Size of population.
        tournament_size(int): Number of competitor individuals per tournament.

    Returns:
        np.array: Winning individual parent chromosome.

    """
    selected_parent = None
    highest_fitness_score = 0.0
    for n in range(tournament_size):
        i = random.randint(0, population_size - 1)
        parent = parent_population[i]
        fitness_score = np.count_nonzero(parent == 1) / parent.shape[0]
        if fitness_score >= highest_fitness_score:
            selected_parent = parent
            highest_fitness_score = fitness_score
    return copy(selected_parent)


def fitness(population):
    """ Determines fitness of individuals in a population.
    """
    population_fitness = np.count_nonzero(population == 1, axis=1) / population.shape[1]
    return population_fitness


def crossover(parents, crossover_rate):
    """ Perform crossover of genes between parent chromosomes.
    """
    offspring = []

    parents = copy(parents)
    parents_indexes = [i for i in range(0, parents.shape[0])]

    while True:
        if not parents_indexes:
            break

        if len(parents_indexes) == 1:
            offspring.append(copy(parents[parents_indexes[0]]))
            break

        parent_index_0 = random.sample(parents_indexes, 1)[0]
        parents_indexes.remove(parent_index_0)
        parent_index_1 = random.sample(parents_indexes, 1)[0]
        parents_indexes.remove(parent_index_1)

        parent_0 = copy(parents[parent_index_0, :])
        parent_1 = copy(parents[parent_index_1, :])

        if random.uniform(0, 1) > crossover_rate:
            offspring += [parent_0, parent_1]
        else:
            cross_over_point = random.randint(0, parents.shape[1] - 2)

            parent_0[cross_over_point:] = parents[parent_index_1][cross_over_point:]
            parent_1[cross_over_point:] = parents[parent_index_0][cross_over_point:]

            offspring += [parent_0, parent_1]

    return np.array(offspring)


def mutate(chromosomes, mutation_probability):
    """ Mutate randomly selected chromosomes.
    """
    for chromosome in chromosomes:
        if random.uniform(0, 1) > mutation_probability:
            continue

        chromosome_length = chromosomes.shape[1]
        mutation_point = random.sample(range(0, chromosome_length - 1), 1)
        mutation_value = chromosome[mutation_point]

        if mutation_value == 0:
            chromosome[mutation_point] = 1
        else:
            chromosome[mutation_point] = 0

    return chromosomes


def next_generation(population, population_size, tournament_size, mutation_rate, crossover_rate):
    """ Generate next generation by selection and applying genetic operators.

    Args:
        population
        population_size
        tournament_size
        mutation_rate
        crossover_rate

    Returns:
        np.array: Next generation.

    """
    selected_parents = []
    for n in range(population_size):
        selected_parent = tournament_selection(population, population_size, tournament_size)
        selected_parents.append(selected_parent)
    selected_parents = np.array(selected_parents)
    next_gen = crossover(selected_parents, crossover_rate)
    next_gen = mutate(next_gen, mutation_rate)
    return next_gen


def evaluate(population, population_fitness):
    for i, fitness_score in enumerate(population_fitness):
        if fitness_score == 1:
            successful_chromosome = population[i]
            return successful_chromosome


def run_genetic_algorithm(
        population_size,
        chromosome_length,
        tournament_size,
        mutation_rate,
        crossover_rate,
        max_generations
):
    """ Run genetic algorithm until convergence or max_generations.

    Args:
        population_size(int): Number of individuals in population.
        chromosome_length(int): Number of genes making up each chromosome.
        tournament_size(int): Number of parents per tournament.
        mutation_rate(float): Probability of individual mutation.
        crossover_rate(float): Probability of crossover between a selected pair of parents.
        max_generations(int): Maximum number of generations to run over.

    """
    population = generate_initial_population(
        population_n=population_size,
        chromosome_length=chromosome_length
    )
    print('========== Start Population ==========')
    print(population)

    population_fitness = fitness(population)

    generation = 0

    while True:
        if np.amax(population_fitness) == 1.0:
            break

        if generation == max_generations:
            break

        generation += 1

        population = next_generation(population, population_size, tournament_size, mutation_rate, crossover_rate)
        population_fitness = fitness(population)
        print(f'========== Generation {generation} ==========')
        print(f'Average population fitness: {sum(population_fitness) / len(population_fitness)}')

    successful_chromosome = evaluate(population, population_fitness)
    print(f'========== Ran For {generation + 1} generations ==========')
    print('Final successful chromosome:')
    print(successful_chromosome)


if __name__ == '__main__':
    run_genetic_algorithm(
        population_size=4,
        chromosome_length=8,
        tournament_size=2,
        mutation_rate=0.2,
        crossover_rate=1.0,
        max_generations=1000
    )
