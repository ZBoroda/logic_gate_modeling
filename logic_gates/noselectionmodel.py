from collections import Counter

import numpy as np

from logic_gates import Circuit

from scipy.spatial.distance import hamming


def run_random_walk(f, N: int, mu: float, x_init: Circuit, t_max: int, m_max: int):
    """
    Run evolutionary dynamics with no selection (random walk).

    Inputs:
      f - fitness landscape, must be function that takes array of length x_init
      N - population size (only affects waiting times, not mutation acceptance)
      mu - mutation rate per site
      x_init - Circuit specifying the initial genotype
      t_max - max simulation time
      m_max - max number of mutations
    """
    x = x_init
    L = len(x_init.genome)
    current_fitness = x_init.evaluate_expression(f)
    # T is kept as a list so len(T) encodes mutation count for the parallel runner
    T = [0.0]
    mutation_counter = Counter()

    while T[-1] < t_max and len(T) < m_max:
        if len(T) % 2000 == 0:
            print(str(len(T)), str(T[-1]), str(current_fitness))
        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        new_fitness = xm.evaluate_expression(f)
        s = new_fitness - current_fitness
        if s == 0:
            mutation_counter["Neutral"] += 1
        elif s > 0:
            mutation_counter["Positive"] += 1
        else:
            mutation_counter["Negative"] += 1
        # accept every mutation (no selection)
        x = xm
        current_fitness = new_fitness
        T.append(T[-1] + tau_next)
        if current_fitness == 1.0:
            break

    F = np.array([x_init.evaluate_expression(f), current_fitness])
    C = np.array([x.num_gates_computing()])
    return np.array(T), x, F, mutation_counter, C, hamming(x.genome, x_init.genome) * len(x.genome)
