import pickle
from logic_gates import run_evolution_strong_selection, run_random_walk, Circuit, IsomorphismCounter, \
    run_in_parallel_same_start
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from tqdm.notebook import tqdm
from goals import and_funct, or_funct, xnor_funct, xor_funct


def construct_genome(size: int) -> list:
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome


def dump_data(data, filename):
    with open('experiment_data/' + filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)


class DataFromExperiment:
    def __init__(self, title, logic_function, network_sizes, num_trials):
        self.fitness_traj = {}
        self.num_computing_traj = {}
        self.fitness_traj_random = {}
        self.num_computing_traj_random = {}
        self.means_mutations = []
        self.medians_mutations = []
        self.stds_mutations = []
        self.means_times = []
        self.medians_times = []
        self.stds_times = []
        self.means_mutations_random = []
        self.medians_mutations_random = []
        self.stds_mutations_random = []
        self.means_times_random = []
        self.medians_times_random = []
        self.stds_times_random = []
        self.isomorphism_counter = IsomorphismCounter()
        self.isomorphism_counter_random = IsomorphismCounter()
        self.mutation_types_list = []
        self.mutation_types_list_random = []
        self.title = title
        self.logic_function = logic_function
        self.sizes = network_sizes
        self.num_trials = num_trials

    def __repr__(self):
        return 'Title: ' + self.title + " for sizes " + self.sizes


def run_experiment(title, logic_function, network_sizes, num_trials):
    data = DataFromExperiment(title, logic_function, network_sizes, num_trials)
    for size in network_sizes:
        print(size)
        initial_circuit = Circuit(2, construct_genome(size))
        mutations, times, mutation_types, fitness_trajs, num_computing_trajs = run_in_parallel_same_start(
            run_evolution_strong_selection, num_trials, size,
            data.isomorphism_counter, logic_function, 1000, 0.01, initial_circuit, 10000, 500000)
        print(fitness_trajs[0][0])
        data.mutation_types_list.append(mutation_types)
        data.fitness_traj[size] = fitness_trajs
        data.num_computing_traj[size] = num_computing_trajs
        times_array = np.array(times)
        data.means_times.append(times_array.mean())
        data.stds_times.append(times_array.std())
        data.medians_times.append(np.median(times_array))
        mutations_array = np.array(mutations)
        data.means_mutations.append(mutations_array.mean())
        data.stds_mutations.append(mutations_array.std())
        data.medians_mutations.append(np.median(mutations_array))
        mutations_random, times_random, mutation_types_random, fitness_trajs_random, num_computing_trajs_random = run_in_parallel_same_start(
            run_evolution_strong_selection, num_trials, size,
            data.isomorphism_counter_random, logic_function, 1000, 0.01, initial_circuit, 10000, 500000)
        data.mutation_types_list_random.append(mutation_types_random)
        data.fitness_traj_random[size] = fitness_trajs_random
        data.num_computing_traj_random[size] = num_computing_trajs_random
        times_array_random = np.array(times_random)
        data.means_times_random.append(times_array_random.mean())
        data.stds_times_random.append(times_array_random.std())
        data.medians_times_random.append(np.median(times_array_random))
        mutations_array_random = np.array(mutations_random)
        data.means_mutations_random.append(mutations_array_random.mean())
        data.stds_mutations_random.append(mutations_array_random.std())
        data.medians_mutations_random.append(np.median(mutations_array_random))
    print("medians times ", data.medians_times, "medians mutations ", data.medians_mutations)
    print("medians times random", data.medians_times_random, "medians mutations_random", data.medians_mutations_random)
    dump_data(data, title)


if __name__ == "__main__":
    # 0 starting fitness
    run_experiment('and', and_funct,
                   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000], 20)
    # 0.25 starting fitness
    run_experiment('xnor', xnor_funct,
                   [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000], 20)
    # 0.5 starting fitness
    run_experiment('or', or_funct,
                   [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000], 20)
    # 0.75 starting fitness
    run_experiment('xor', xor_funct,
                   [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000], 20)
