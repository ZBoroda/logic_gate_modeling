"""
Fast simulation models using FastCircuit.
Drop-in replacements for run_random_walk and run_evolution_strong_selection.
Accepts Circuit or FastCircuit as x_init; always returns the same 6-tuple.
"""
import numpy as np
from collections import Counter
from scipy.spatial.distance import hamming

from fast_circuit.circuit import FastCircuit


def _to_fast(x_init) -> FastCircuit:
    """Accept either a Circuit or FastCircuit and return a FastCircuit."""
    if isinstance(x_init, FastCircuit):
        return x_init.duplicate()
    return FastCircuit(x_init.genome)


def run_random_walk_fast(f, N: int, mu: float, x_init, t_max: int, m_max: int):
    x = _to_fast(x_init)
    init_genome = x.genome.copy()
    L = len(x.genome)
    current_fitness = x.fitness(f)
    T = [0.0]
    mutation_counter = Counter()

    while T[-1] < t_max and len(T) < m_max:
        if len(T) % 2000 == 0 and len(T) > 0:
            print(str(len(T)), str(T[-1]), str(current_fitness))

        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        new_fitness = xm.fitness(f)

        s = new_fitness - current_fitness
        if s == 0:
            mutation_counter["Neutral"] += 1
        elif s > 0:
            mutation_counter["Positive"] += 1
        else:
            mutation_counter["Negative"] += 1

        x = xm
        current_fitness = new_fitness
        T.append(T[-1] + tau_next)

        if current_fitness == 1.0:
            break

    F = np.array([x_init.fitness(f) if isinstance(x_init, FastCircuit)
                  else 0.0,           # original Circuit doesn't have .fitness()
                  current_fitness])
    C = np.array([0])  # computing-edge count not tracked in fast model
    return (np.array(T), x, F, mutation_counter, C,
            hamming(x.genome, init_genome) * len(x.genome))


def run_strong_selection_fast(f, N: int, mu: float, x_init, t_max: int, m_max: int):
    x = _to_fast(x_init)
    init_genome = x.genome.copy()
    L = len(x.genome)
    current_fitness = x.fitness(f)
    T = [0.0]
    mutation_counter = Counter()

    while T[-1] < t_max and len(T) < m_max:
        if len(T) % 10000 == 0 and len(T) > 0:
            print('\t', str(len(T)), str(T[-1]), str(current_fitness))

        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        new_fitness = xm.fitness(f)

        s = new_fitness - current_fitness
        if s != 0:
            if abs(-s * N) < 250:
                pfix = (1 - np.exp(-s)) / (1 - np.exp(-s * N))
            elif -s * N > 0:
                pfix = 0.0
            else:
                pfix = 1 - np.exp(-s)
        else:
            pfix = 1 / N

        if np.random.rand() < pfix:
            x = xm
            current_fitness = new_fitness
            if s == 0:
                mutation_counter["Neutral"] += 1
            elif s > 0:
                mutation_counter["Positive"] += 1
            else:
                mutation_counter["Negative"] += 1

        T.append(T[-1] + tau_next)

        if current_fitness == 1.0:
            break

    F = np.array([0.0, current_fitness])
    C = np.array([0])
    return (np.array(T), x, F, mutation_counter, C,
            hamming(x.genome, init_genome) * len(x.genome))
