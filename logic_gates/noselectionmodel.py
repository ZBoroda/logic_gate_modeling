from collections import Counter

import numpy as np

from logic_gates import Circuit

from scipy.spatial.distance import hamming


def run_random_walk(f, N: int, mu: float, x_init: Circuit, t_max: int, m_max: int):
    """
    Run evolutionary dynanamics in strong selection weak mutation regime.

    Inputs:
      f - fitness landscape, must be function that takes array of length x_init
      N - population size
      mu - mutation rate per site
      x_init - array of length L specifying the initial genome
      t_max - the time to run the simulation

    """
    p = 0
    n = 0
    T = [0.0]
    C = []  # C is the number of computing nodes
    F = [x_init.evaluate_expression(f)]
    X = [x_init]
    x = x_init
    L = len(x_init.genome)
    mutation_counter = Counter()
    while T[-1] < t_max and len(T) < m_max:
        if len(T) % 2000 == 0:
            print(str(len(T)), str(T[-1]), str(F[-1]))
        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        old_fitness = x.evaluate_expression(f)
        new_fitness = xm.evaluate_expression(f)
        s = (new_fitness - old_fitness)
        if s == 0:
            mutation_counter["Neutral"] += 1
        elif s > 0:
            mutation_counter["Positive"] += 1
        elif s < 0:
            mutation_counter["Negative"] += 1
        x = xm.duplicate()
        # update
        C.append(x.num_gates_computing())
        F.append(x.evaluate_expression(f))
        T.append(T[-1] + tau_next)
        if F[-1] == 1.0:
            break
    return np.array(T), x, np.array(F), mutation_counter, np.array(C), hamming(x.genome, x_init.genome) * len(x.genome)
