'''
First imports
'''
import pickle

from logic_gates import run_evolution_strong_selection, run_random_walk, Circuit
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
from tqdm.notebook import tqdm
import concurrent.futures
from goals import and_funct
import time


def construct_genome(size: int) -> list:
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome


class IsomorphismCounter:

    def __init__(self, counter = None):
        if counter == None:
            self.counter = defaultdict(Counter)
        else:
            self.counter = counter

    def add(self, network, size: int):
        for key in self.counter:
            if key.is_isomorphic(network, pruned=True):
                self.counter[key][size] += 1
                self.counter[key]["total"] += 1
                return key
        self.counter[network][size] += 1
        self.counter[network]["total"] += 1
        return network

    def get_networks(self):
        return self.counter.keys()

    def get_number_by_size(self, size: int):
        return {key: value[size] for key, value in self.counter.items()}

    def get_number_networks_total(self):
        return {key: value['total'] for key, value in self.counter.items()}


def run_in_parallel(function, axis, num_itter, size, isomorphism_counter, *args):
    total_times = []
    total_mutations = []
    total_mutations_counter = Counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = [executor.submit(function, *args) for _ in range(num_itter)]
        for f in concurrent.futures.as_completed(future_results):
            mutationtimes, circuit, fitness, mutation_counter = f.result()
            axis.plot(fitness)
            if fitness[-1] == 1:
                total_mutations.append(len(mutationtimes))
                total_times.append(mutationtimes[-1])
                isomorphism_counter.add(circuit, size)
                total_mutations_counter += mutation_counter
            else:
                print("uh oh didnt get to full fitness")
        return total_mutations, total_times, {key: value / num_itter for key, value in total_mutations_counter.items()}

def dump_data(data, filename):
    with open('andlargesize/' + filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000]
    fig_trajectory, ax_trajectories = plt.subplots(len(sizes), 1, sharex=True, sharey=True, figsize=(9, 20))
    fig_histogram_mutations, ax_histograms_mutations = plt.subplots(len(sizes), 1, sharex=True, sharey=True,
                                                                    figsize=(9, 20))
    fig_histogram_times, ax_histograms_times = plt.subplots(len(sizes), 1, sharex=True, sharey=True, figsize=(9, 20))
    fig_trajectory_random, ax_trajectories_random = plt.subplots(len(sizes), 1, sharex=True, sharey=True,
                                                                 figsize=(9, 20))
    fig_histogram_random, ax_histograms_random = plt.subplots(len(sizes), 1, sharex=True, sharey=True, figsize=(9, 20))
    means_mutations = []
    medians_mutations = []
    stds_mutations = []
    means_times = []
    medians_times = []
    stds_times = []
    means_mutations_random = []
    medians_mutations_random = []
    stds_mutations_random = []
    means_times_random = []
    medians_times_random = []
    stds_times_random = []
    isomorphism_counter = IsomorphismCounter()
    isomorphism_counter_random = IsomorphismCounter()
    mutation_types_list = []
    mutation_types_list_random = []
    num_trials = 100
    for size, ax_trajectory, ax_histogram_mutation, ax_histogram_times, ax_trajectory_random, ax_histogram_random in \
            zip(
            sizes, ax_trajectories,
            ax_histograms_mutations,
            ax_histograms_times,
            ax_trajectories_random,
            ax_histograms_random):
        ax_trajectory.set_title(size)
        ax_histogram_mutation.set_title(size)
        ax_histogram_times.set_title(size)
        ax_trajectory_random.set_title(size)
        ax_histogram_random.set_title(size)
        initial_circuit = Circuit(2, construct_genome(size))
        print(size)
        mutations, times, mutation_types = run_in_parallel(run_evolution_strong_selection, ax_trajectory, num_trials, size,
                                           isomorphism_counter, and_funct, 1000, 0.1, initial_circuit, 10000, 500000)
        mutation_types_list.append(mutation_types)
        times_array = np.array(times)
        mutations_array = np.array(mutations)
        medians_mutations.append(np.median(mutations_array))
        medians_times.append(np.median(times_array))
        print("Median mutations:" + str(medians_mutations[-1]))
        means_mutations.append(mutations_array.mean())
        means_times.append(times_array.mean())
        print("Mean mutations:" + str(means_mutations[-1]))
        stds_mutations.append(mutations_array.std())
        stds_times.append(times_array.std())
        print("STD mutations:" + str(stds_mutations[-1]))
        ax_histogram_mutation.hist(mutations, 20)
        ax_histogram_times.hist(times, 20)
        mutations_random, times_random, mutation_types_random = run_in_parallel(run_random_walk, ax_trajectory_random, num_trials, size,
                                           isomorphism_counter_random, and_funct, 1000, 0.1, initial_circuit, 10000, 500000)
        mutation_types_list_random.append(mutation_types_random)
        random_mutations_array = np.array(mutations_random)
        random_times_array = np.array(times_random)
        medians_mutations_random.append(np.median(random_mutations_array))
        medians_times_random.append(np.median(random_times_array))
        print("Median mutations random:" + str(medians_mutations[-1]))
        means_mutations_random.append(random_mutations_array.mean())
        means_times_random.append(random_times_array.mean())
        print("Mean mutations random:" + str(means_mutations[-1]))
        stds_mutations_random.append(random_mutations_array.std())
        stds_times_random.append(random_times_array.std())
        print("STD mutations random:" + str(stds_mutations[-1]))
        ax_histogram_random.hist(mutations_random, 20)

    dump_data(isomorphism_counter.counter, "isomorphism_counter_evo")
    dump_data(isomorphism_counter_random.counter, "isomorphism_counter_random")
    dump_data(mutation_types_list, "types_of_mutations")
    dump_data(mutation_types_list_random, "types_of_mutations_random")
    dump_data(means_mutations, "means_mutations")
    dump_data(stds_mutations, "stds_mutations")
    dump_data(medians_mutations, "medians_mutations")
    dump_data(means_times, "means_times")
    dump_data(stds_times, "stds_times")
    dump_data(medians_times, "medians_times")
    dump_data(means_mutations_random, "means_mutations_random")
    dump_data(stds_mutations_random, "stds_mutations_random")
    dump_data(medians_mutations_random, "medians_mutations_random")
    dump_data(means_times_random, "means_times_random")
    dump_data(stds_times_random, "stds_times_random")
    dump_data(medians_times_random, "medians_times_random")
    dump_data(sizes, "sizes")


    fig_trajectory.savefig('trajectories_and_big.png', dpi=1200)
    fig_histogram_mutations.suptitle("Histogram of Number of Mutations")
    fig_histogram_mutations.savefig('histogram_fixations_mutations_and_big.png')
    fig_histogram_times.suptitle("Histogram of Fixation Times")
    fig_histogram_times.savefig('histogram_fixations_times_and_big.png')
    plt.show()

    fig_trajectory_random.savefig('trajectories_and_big_random.png', dpi=1200)
    fig_histogram_random.suptitle("Histogram of Number of Mutations Random Walk")
    fig_histogram_random.savefig('histogram_fixations_mutations_and_big_random.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_mutations, yerr=np.array(stds_mutations) / np.sqrt(1000))
    ax.set_title("Mean Number of Mutations Until Fixation")
    fig.savefig('meanmutationstillfixationAnd.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_mutations)
    ax.set_title("Median Number of Mutations Until Fixation")
    fig.savefig('medianmutationstillfixationAnd.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times, yerr=np.array(stds_times) / np.sqrt(1000))
    ax.set_title("Mean Time Until Fixation")
    fig.savefig('meantimetillfixationAnd.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_times)
    ax.set_title("Median Time Until Fixation")
    fig.savefig('mediantimetillfixationAnd.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_mutations_random, yerr=np.array(stds_mutations_random) / np.sqrt(1000))
    ax.set_title("Mean Number of Mutations Until Fixation Random Walk")
    fig.savefig('meanmutationstillfixationAndRandom.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_mutations_random)
    ax.set_title("Median Number of Mutations Until Fixation Random Walk")
    fig.savefig('medianmutationstillfixationAndRandom.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times_random, yerr=np.array(stds_times_random) / np.sqrt(1000))
    ax.set_title("Mean Time Until Fixation Random Walk")
    fig.savefig('meantimetillfixationAndRandom.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times_random)
    ax.set_title("Median Time Until Fixation Random Walk")
    fig.savefig('mediantimetillfixationAndRandom.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_mutations, yerr=np.array(stds_mutations) / np.sqrt(1000))
    ax.set_yscale("log")
    ax.set_title("Mean Number of Mutations Until Fixation LogY")
    fig.savefig('meanmutationstillfixationAndLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_mutations)
    ax.set_yscale("log")
    ax.set_title("Median Number of Mutations Until Fixation LogY")
    fig.savefig('medianmutationstillfixationAndLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times, yerr=np.array(stds_times) / np.sqrt(1000))
    ax.set_yscale("log")
    ax.set_title("Mean Time Until Fixation LogY")
    fig.savefig('meantimetillfixationAndLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_times)
    ax.set_yscale("log")
    ax.set_title("Median Time Until Fixation LogY")
    fig.savefig('mediantimetillfixationAndLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_mutations_random, yerr=np.array(stds_mutations_random) / np.sqrt(1000))
    ax.set_yscale("log")
    ax.set_title("Mean Number of Mutations Until Fixation Random Walk LogY")
    fig.savefig('meanmutationstillfixationAndRandomLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=medians_mutations_random)
    ax.set_yscale("log")
    ax.set_title("Median Number of Mutations Until Fixation Random Walk LogY")
    fig.savefig('medianmutationstillfixationAndRandomLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times_random, yerr=np.array(stds_times_random) / np.sqrt(1000))
    ax.set_yscale("log")
    ax.set_title("Mean Time Until Fixation Random Walk LogY")
    fig.savefig('meantimetillfixationAndRandomLogY.png', dpi=1200)
    plt.show()

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.errorbar(x=sizes, y=means_times_random)
    ax.set_yscale("log")
    ax.set_title("Median Time Until Fixation Random Walk LogY")
    fig.savefig('mediantimetillfixationAndRandomLogY.png', dpi=1200)
    plt.show()