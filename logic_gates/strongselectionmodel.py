import numpy as np
import random

import numpy as np

from logic_gates import Circuit


def run_evo(f, N: int, mu: float, x_init: Circuit, t_max: int):
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
    F = [x_init.evaluate_expression(f)]
    X = [x_init]
    x = x_init
    L = len(x_init.genome)
    neutral = 0
    neutral_fix = 0
    while T[-1] < t_max:
        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        old_fitness = x.evaluate_expression(f)
        new_fitness = xm.evaluate_expression(f)
        s = (new_fitness - old_fitness)  # for turbidostat we would use (f(xm)-f(x))/f(x)
        # the formula holds for all s but we have numerical problems when s is too small
        # there are better ways around this
        if s != 0:
            if np.abs(-s * N) < 250:
                pfix = (1 - np.exp(-s)) / (1 - np.exp(-s * N))
            elif -s * N > 0:
                pfix = 0
            else:
                pfix = 1 - np.exp(-s)
        else:
            pfix = 1 / N
        if s == 0:
            neutral += 1
        r = np.random.rand()
        if r < pfix:
            x = xm.duplicate()
            if s == 0:
                neutral_fix += 1
        # update
        F.append(x.evaluate_expression(f))
        X.append(x)
        T.append(T[-1] + tau_next)
        if F[-1] == 1.0:
            break
    return np.array(T), np.array(X), np.array(F)
