from logic_gates import Circuit
from logic_gates import FullModel
from multiprocessing import Pool
import os
import sys
import time

import networkx as nx
import numpy as np
import random

import matplotlib.pyplot as plt

# First let us define our goal function

def goal_function(x):
    return x[0] and x[1]

class SimulationParallelizer:

    def __init__(self, number_iterations, population_size, num_input_nodes, genome_size, mutation_probability,
                 mutation_proportional_to_size, function, max_gens):
        self.number_iterations = number_iterations
        self.population_size = population_size
        self.num_input_nodes = num_input_nodes
        self.genome_size = genome_size
        self.function = function
        self.mutation_probability = mutation_probability
        self.mutation_proportional_to_size = mutation_proportional_to_size
        self.max_gens = max_gens

    def run_simulation(self):
        with Pool() as pool:
            results = pool.map(self.run_one, [i for i in range(self.number_iterations)])
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        time_till_fixation = [result['First Time Full Fitness'] for result in results]
        print(time_till_fixation)
        ax.hist(time_till_fixation, 100)
        ax.set_title("Histogram of Fixation Time, Genome Size: " + str(self.genome_size))
        fig.savefig('histogram_fixation' + str(self.genome_size) + '.pdf', dpi=1200)
        return time_till_fixation

    def run_one(self, cnt):
        genome = []
        for i in range(self.genome_size):
            genome += [0, 1]
        genome += [2]
        model = FullModel(self.population_size, self.num_input_nodes, genome,
                          mutation_probability=self.mutation_probability,
                          mutation_proportional_to_size=self.mutation_proportional_to_size)
        fitness, circuits, mean_fitness, first_full_fitness = model.evolve(self.function, max_gens=self.max_gens,
                                                                           debug=False)
        return {'Max Fitness': fitness,
                'Circuits': circuits,
                "Mean Fitness": mean_fitness,
                "First Time Full Fitness": first_full_fitness,
                "Number": cnt}


'''


def run_one_size(size):
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]


def run_one_genome(genome):
    model = Model(1000, 2, genome, mutation_probability=0.001)
    fitness, _, mean_fitness, first_full_fitness = model.evolve(goal_function, max_gens=8000, debug=False)
    return fitness, mean_fitness, first_full_fitness

'''

if __name__ == "__main__":
    '''
    sys.path.append(os.getcwd())
    time_fixations_mean = []
    time_fixations_std = []
    range_sizes = list(range(2, 15))
    mutation_proportional_to_size = True
    for genome_size in range_sizes:
        parallelizer = SimulationParallelizer(192, 1000, 2, genome_size, 0.001, mutation_proportional_to_size,
                                              goal_function, 20000)
        time_fixation = np.array(parallelizer.run_simulation())
        time_fixations_mean.append(time_fixation.mean())
        time_fixations_std.append(time_fixation.std())
        print(genome_size)
        print('\t' + str(time_fixation.mean()))
        print('\t' + str(time_fixation.std()))
        print('\t' + str(time_fixation.var()))
        print('\t' + str(time_fixation))
    fig, ax = plt.subplots()
    ax.set_title("Box Plot of Fixation Time By Size")
    ax.errorbar(x=range_sizes, y=time_fixations_mean, yerr=time_fixations_std)
    if mutation_proportional_to_size:
        fig.savefig('fixation_by_size_mutpropsize' + '.png', dpi=1200)
    else:
        fig.savefig('fixation_by_size_notmutpropsize' + '.png', dpi=1200)
    '''
    # Now lets create a model object with a population size of 1000
    m = FullModel(10 ** 3, 2, [0, 1, 0, 1, 0, 1, 2], mutation_probability=10 ** (-3))
    fitness, circuits, mean_fitness, first_full_fitness = m.evolve(goal_function, max_gens=1000)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(fitness)
    ax[1].plot(mean_fitness)
    ax[0].set_ylabel('Top fitness')
    ax[1].set_ylabel('Mean fitness')
    ax[1].set_xlabel('Generation')
    fig.savefig('evolving_and.pdf', dpi=1200)

    # Now lets plot the top circuit
    # First the full thing
    circuits[0].plot_network(prune=False, filename='unpruned_and.pdf')

    # Now just the pruned network
    circuits[0].plot_network(prune=True, filename='pruned_and.pdf')
    '''
    possible_sizes = [2, 3, 4, 5, 6, 7, 8]
    number_iterations = 10
    mean_tt_full_fitness = []
    mean_tt_fixation = []
    for size in possible_sizes:
        genome = []
        for i in range(size):
            genome += [0, 1]
        genome += [2]
        tt_fitness = []
        tt_fixation = []
        for interation in range(number_iterations):
            m = Model(1000, 2, genome, mutation_probability=0.001)
            fitness, _, mean_fitness, first_full_fitness = m.evolve(goal_function, max_gens=8000, debug=False)
            tt_fitness.append(first_full_fitness)
            tt_fixation.append(len(mean_fitness))
        mean_tt_full_fitness.append(float(sum(tt_fitness)) / number_iterations)
        mean_tt_fixation.append(float(sum(tt_fixation)) / number_iterations)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(possible_sizes, mean_tt_full_fitness)
    ax[1].plot(possible_sizes, mean_tt_fixation)
    ax[0].set_ylabel('Generations until 1.0')
    ax[1].set_ylabel('Generations until 1.0 fixates')
    ax[1].set_xlabel('Number of Gates')
    fig.savefig('size_vs_fixation_time.pdf', dpi=1200)
    '''