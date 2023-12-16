import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# This function calculates the influence of a dominator in a given position within the matrix
# It iterates through the matrix and adds to the influence count for each cell within the max_distance from the dominator
def calculate_influence(place_of_dominator, matrix, max_distance):
    rows, columns = matrix.shape
    influence = 0
    for i in range(rows):
        for j in range(columns):
            if matrix[i][j] == 1:
                distance = max(abs(place_of_dominator[0] - i), abs(place_of_dominator[1] - j))
                if distance <= max_distance:
                    influence += 1
    return influence

# This function generates an initial population for the genetic algorithm
# It creates a list of unique matrices, each representing a potential solution
def generate_initial_population(population_size, n):
    # Permutation used for uniqueness of matrices
    return [np.array([np.random.permutation(np.eye(1, n, k).flatten()) for k in range(n)]) for _ in range(population_size)]

# This function calculates the fitness of a given dominator configuration in the matrix
# The fitness is a measure of how effective a particular arrangement of dominators is
def calculate_fitness(solution, matrix, strategic_values, risk_assessment, max_distance):
    total_influence = 0
    for i in range(len(solution)):
        for j in range(len(solution[0])):
            if solution[i][j] == 1:
                dominator_influence = calculate_influence((i, j), matrix, max_distance)
                total_influence += dominator_influence
    penalty = 0
    for i in range(matrix.shape[0]):
        if np.sum(solution[i, :]) != 1 or np.sum(solution[:, i]) != 1:
            penalty += 10
    fitness = total_influence - penalty
    return fitness

# This function sorts the population based on their fitness scores and selects the top ones
def rank_selection(population, fitness_scores, population_size):
    sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:population_size]

# This function randomly selects groups of individuals and chooses the best from each group to form a new population
def tournament_selection(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        best_participant = max(participants, key=lambda idx: fitness_scores[idx])
        selected_parents.append(population[best_participant])
    return selected_parents

# Function to determine if a dominator is needed at a specific position in the matrix
# It checks the surrounding cells to ensure there's no dominator within a one-cell range
def is_dominator_needed(individual, i, j, n):
    for ii in range(-1, 2):
        for jj in range(-1, 2):
            if 0 <= i + ii < n and 0 <= j + jj < n and individual[i + ii, j + jj] == 1:
                return False
    return True

# Function to enforce constraints on the placement of dominators in the matrix
# It ensures that no more than one dominator is in one row or column at any time
def apply_constraints(individual, n):
    n = individual.shape[0]

    # Check if a position in the matrix is valid for placing a dominator
    def is_valid_position(ind, i, j, n):
        for col in range(n):
            if col != j and ind[i, col] == 1:
                return False
        for row in range(n):
            if row != i and ind[row, j] == 1:
                return False
        return True

    # Place a dominator in a valid row position
    def place_dominator_in_row(ind, i, n):
        n = ind.shape[1]
        attempts = 0
        max_attempts = 10
        while attempts < max_attempts:
            random_position = np.random.randint(n)
            if is_valid_position(ind, i, random_position, n):
                ind[i, :] = 0
                ind[i, random_position] = 1
                return
            attempts += 1

    # Place a dominator in a valid column position
    def place_dominator_in_column(ind, j):
        spots = [i for i in range(n) if is_valid_position(ind, i, j, n)]
        if spots:
            take_position = random.choice(spots)
        else:
            take_position = np.random.randint(n)
        ind[:, j] = 0
        ind[take_position, j] = 1

    # Apply constraints for each row and column
    for i in range(n):
        if np.sum(individual[i, :]) != 1 or not all(is_valid_position(individual, i, j, n) for j in range(n) if individual[i, j] == 1):
            place_dominator_in_row(individual, i, n)
    for j in range(n):
        if np.sum(individual[:, j]) != 1 or not all(is_valid_position(individual, dominator_place, j, n) for dominator_place in range(n) if individual[dominator_place, j] == 1):
            place_dominator_in_column(individual, j)
    return individual

# Function to perform single point crossover between two parent matrices
# It creates two new matrices (offspring) by swapping parts of the parent matrices at a randomly chosen point
def crossover(parent1, parent2):
    rows, columns = parent1.shape
    first_output, second_output = parent1.copy(), parent2.copy()
    point = np.random.randint(1, columns)
    first_output[:, point:], second_output[:, point:] = second_output[:, point:], first_output[:, point:]

    first_output = apply_constraints(first_output, n)
    second_output = apply_constraints(second_output, n)

    return first_output, second_output

# Function to introduce mutations within the matrices
# It randomly flips the dominator's state in the matrix based on a given mutation rate
def mutation(individual, mutation_rate, n):
    for i in range(n):
        for j in range(n):
            if random.random() < mutation_rate:
                individual[i, j] = 1 - individual[i, j]
    apply_constraints(individual, n)
    return individual

# Function to select the best solutions based on their fitness scores
# It sorts the population by fitness and retains the top-performing individuals
def elitism(population, fitness_scores, elite_size):
    sorted_by_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    elites = [individual for individual, _ in sorted_by_fitness[:elite_size]]
    return elites

# Function to apply the wisdom of crowds principle in selecting the next generation of solutions
# It aggregates the decisions of the population to form a consensus solution
def swarm_of_wisdom(population):
    number_of_rows, number_of_columns = population[0].shape
    what_is_wise = np.zeros((number_of_rows, number_of_columns))

    for i in range(number_of_rows):
        for j in range(number_of_columns):
            values = [individual[i, j] for individual in population]
            most_common_value = max(set(values), key=values.count)
            what_is_wise[i, j] = most_common_value

    calculate_difference = [np.sum(individual == what_is_wise) for individual in population]
    sorted = np.argsort(calculate_difference)[::-1]
    swarm_selection = [population[idx] for idx in sorted[:len(population)//2]]
    return swarm_selection

# Main function to run the trial of the genetic algorithm
# It generates a specified number of random matrices and applies the genetic algorithm to find the best solutions
# It returns a list of tuples, each containing the original and the best solution found for each matrix
def trial(number_of_matrices, n, population_size, number_of_generations, max_dominators, mutation_rate):
    original_and_best_solutions = []
    for i in range(number_of_matrices):
        print(f"Running Matrix #{i + 1}/{number_of_matrices}")
        random_matrix = np.random.randint(2, size=(n, n))
        best_solution = genetic_algorithm(random_matrix, population_size, number_of_generations, max_dominators, mutation_rate)
        if best_solution is not None:
            original_and_best_solutions.append((random_matrix, best_solution))
        else:
            print(f"Error in finding a solution for Matrix #{i + 1}")

    return original_and_best_solutions

# Function to visually display the original and best solution matrices using matplotlib
# It creates a grid of subplots to compare the original matrices with their best solutions side by side
def display_matrices(original_and_best_solutions):
    num_matrices = len(original_and_best_solutions)
    num_subplots = num_matrices * 2
    columns = 5
    rows = num_subplots // columns + (1 if num_subplots % columns else 0)
    fig_width = 20
    fig_height = rows * 3
    plt.rcParams['font.family'] = 'Times New Roman'
    cmap = mcolors.ListedColormap(['#0c0605', '#0dabd9'])

    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
    fig.suptitle("Matrix Domination\n(blue indicates dominator, black indicates dominated)", fontsize=12)

    subplot_index = 0
    for i, (original, best_solution) in enumerate(original_and_best_solutions):
        row = subplot_index // columns
        col = subplot_index % columns

        axes[row, col].imshow(original, cmap=cmap)
        axes[row, col].set_title(f'Matrix {i+1}:\n Original', fontsize=6)
        axes[row, col].axis('off')
        subplot_index += 1

        row = subplot_index // columns
        col = subplot_index % columns
        axes[row, col].imshow(best_solution, cmap=cmap)
        axes[row, col].set_title(f'Matrix {i+1}:\n Solution', fontsize=6)
        axes[row, col].axis('off')
        subplot_index += 1

    for j in range(subplot_index, rows * columns):
        axes[j // columns, j % columns].axis('off')

    plt.subplots_adjust(wspace=.38, hspace=.1)
    plt.show()

# Main function to implement the genetic algorithm for solving the matrix domination problem
# This function synthesizes all the previously defined operations like selection, crossover, mutation, and constraint application
# It returns the best individual matrix solution found across all generations
def genetic_algorithm(matrix, population_size, number_of_generations, max_dominators, mutation_rate):
    n = matrix.shape[0]
    constraints = {'max_dominators': max_dominators}
    strategic_values = np.ones((n, n))
    risk_assessment = np.ones((n, n))
    max_distance = n
    population = [apply_constraints(np.random.randint(2, size=(n, n)), n) for _ in range(population_size)]

    for generation in range(number_of_generations):
        fitness_scores = [calculate_fitness(ind, matrix, strategic_values, risk_assessment, max_distance) for ind in population]
        if not fitness_scores:
            print("ERROR.")
            break

        selected = rank_selection(population, fitness_scores, population_size)
        if not selected:
            print("ERROR.")
            break
        selected = swarm_of_wisdom(population)

        new_population = []

        while len(new_population) < population_size:
            inputs = np.random.choice(len(selected), 2, replace=False)
            first_input, second_input = selected[inputs[0]], selected[inputs[1]]
            first_output, second_output = crossover(first_input, second_input)

            first_output = mutation(first_output, mutation_rate, n)
            second_output = mutation(second_output, mutation_rate, n)
            first_output = apply_constraints(first_output, n)
            second_output = apply_constraints(second_output, n)
            new_population.extend([first_output, second_output])

        population = new_population[:population_size]
        print(f"Generation {generation + 1}")

    fitness_scores = [calculate_fitness(ind, matrix, strategic_values, risk_assessment, max_distance) for ind in population]
    if not fitness_scores:
        print("ERROR.")
        return None

    best_individual_index = np.argmax(fitness_scores)
    best_individual = population[best_individual_index]

    return best_individual

# Build and run the algorithm
n = 8  # Size of the matrix
K = n  # Max number of dominators (set to the size of the matrix to fit within NP-Complete problem definition)
population_size = 100  # Number of individuals in each generation
number_of_generations = 25  # Total number of generations for evolutionary algorithm to run
number_of_matrices = 5  # Number of matrices to solve
mutation_rate = 0.04  # Probability of mutation

# Run the genetic algorithm
original_and_best_solutions = trial(number_of_matrices, n, population_size, number_of_generations, K, mutation_rate)

# Numerical console display of the original and best solutions for each matrix
print("\n==================== Genetic Algorithm Results ====================")
for i, (original, best_solution) in enumerate(original_and_best_solutions):
    print("\nOriginal Matrix:")
    print(original)
    print("\nOptimized Solution:")
    print(best_solution)

# Visual display of the matrices using matplotlib for results
display_matrices(original_and_best_solutions)