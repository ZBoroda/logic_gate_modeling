"""
Tests for faster_circuit.

  1. Fitness agrees with FastCircuit on random DAGs (all 16 target functions).
  2. Local cycle test agrees with FastCircuit's global topological sort.
  3. Parallel start, N_P = 1e4: simulated <M> and first-step mean agree with
     the exact formulas of the bottleneck note (Sec. 2) within 4 SEM.
  4. Speed benchmark vs FastCircuit (informational).
"""
import math
import random
import time

import numpy as np

from fast_circuit import FastCircuit
from fast_circuit.circuit import warmup as warmup_fast
from faster_circuit import (fitness, has_cycle_via, run_strong_selection_faster,
                            pfix_table, warmup)
from goals import and_funct


def random_dag_genome(n_gates, rng):
    """Random acyclic genome with gate labels shuffled (not in topological order)."""
    perm = list(range(n_gates))
    rng.shuffle(perm)                       # perm[k] = label of k-th gate in topo order
    genome = [0] * (2 * n_gates + 1)
    for k, label in enumerate(perm):
        earlier = [0, 1] + [perm[j] + 2 for j in range(k)]
        genome[2 * label] = rng.choice(earlier)
        genome[2 * label + 1] = rng.choice(earlier)
    genome[-1] = rng.randrange(n_gates) + 2
    return genome


def all_functions():
    for t in range(16):
        yield t, (lambda x, t=t: bool((t >> (2 * x[0] + x[1])) & 1))


def test_fitness_matches_fast_circuit():
    rng = random.Random(1)
    for _ in range(300):
        g = random_dag_genome(rng.randint(2, 40), rng)
        fc = FastCircuit(g)
        for _, f in all_functions():
            assert fitness(g, f) == fc.fitness(f), (g,)
    print("  fitness: OK (300 random DAGs x 16 targets)")


def test_cycle_test_matches_topological_sort():
    rng = random.Random(2)
    n_checked = 0
    for _ in range(300):
        n = rng.randint(2, 30)
        g = random_dag_genome(n, rng)
        for _ in range(20):
            gate = rng.randrange(n)
            slot = rng.randrange(2)
            src = rng.randrange(n + 2)
            if src == gate + 2:
                continue
            g2 = list(g)
            g2[2 * gate + slot] = src
            assert has_cycle_via(g, gate + 2, src) == FastCircuit(g2).has_cycle()
            n_checked += 1
    print(f"  cycle test: OK ({n_checked} proposals)")


def exact_parallel(N, Np):
    """Exact <M> and 1/p0 for the parallel start (bottleneck note, Eq. 3)."""
    pf = pfix_table(Np)
    a, b = pf[2 + 4], pf[1 + 4]
    p0 = 2 * ((N - 1) * a + b) / (N * (2 * N + 1))
    M = ((2 * N + 1) * N ** 2 / (2 * (N - 1) * ((N - 1) * a + b))
         + ((2 * N + 1) * N - 2) / ((N - 1) * a))
    return M, 1 / p0


def parallel_genome(N):
    return [0, 1] * N + [2]


def test_parallel_start_matches_exact():
    for N, trials in [(10, 20000), (50, 8000)]:
        M, counts = run_strong_selection_faster(and_funct, 1e4, parallel_genome(N),
                                                n_trials=trials, seed=123)
        M_ex, first_ex = exact_parallel(N, 1e4)
        for name, x, ex in [("<M>", M, M_ex), ("first step", counts[:, 0], first_ex)]:
            mean, sem = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
            z = (mean - ex) / sem
            print(f"  N={N:3d} {name:10s}: sim {mean:8.2f} ± {sem:5.2f}, exact {ex:8.2f}, z={z:+.2f}")
            assert abs(z) < 4


def test_speed():
    for N in [100, 1000, 10000]:
        g = parallel_genome(N)
        t = time.perf_counter()
        M, _ = run_strong_selection_faster(and_funct, 1e4, g, n_trials=20, seed=0)
        dt = time.perf_counter() - t
        print(f"  faster_circuit N={N:5d}: {dt / M.sum() * 1e6:6.3f} µs/mutation "
              f"({dt / 20 * 1e3:7.1f} ms/run)")
    for N in [100, 1000, 10000]:
        x = FastCircuit(parallel_genome(N))
        K = 2000
        t = time.perf_counter()
        for _ in range(K):
            xm = x.duplicate(); xm.mutate(); xm.fitness(and_funct)
        print(f"  fast_circuit   N={N:5d}: {(time.perf_counter() - t) / K * 1e6:6.3f} µs/mutation")


if __name__ == "__main__":
    warmup(); warmup_fast()
    test_fitness_matches_fast_circuit()
    test_cycle_test_matches_topological_sort()
    test_parallel_start_matches_exact()
    test_speed()
    print("All tests passed.")
