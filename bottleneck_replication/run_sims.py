"""
Simulate the bottleneck note's experiments with both engines:

  faster : faster_circuit (note's algorithm, Methods mutation distribution)
  fast   : fast_circuit.FastCircuit, using exactly the loop of
           fast_circuit.models.run_strong_selection_fast (same RNG calls),
           instrumented to count mutations spent at each fitness level.

Families (N_P = 1e4, target AND):
  alpha = 0      parallel start,  N = 5 .. 10^4
  alpha in A     interpolating start (L ~ N^alpha), N = 100 .. 10^4
  alpha = 1      chain start, odd N = 5 .. 10001

Results go to bottleneck_replication/results/<engine>_a<alpha>_N<N>.npz with
arrays M (mutations per run), counts (runs x 5, mutations at k correct rows),
and seconds (wall time of the whole config). Existing files are skipped, so
the script can be interrupted and resumed.

Usage:  python -m bottleneck_replication.run_sims [faster|fast|both] [n_workers]
"""
import os
import sys
import time
import random
from multiprocessing import Pool

import numpy as np

from bottleneck_replication.starts import interpolating, chain, module_length

NP = 1e4
OUT = os.path.join(os.path.dirname(__file__), 'results')
M_MAX = 10_000_000

ALPHAS_MID = [0.2, 0.4, 0.5, 0.6, 0.75, 0.9]
N_PARALLEL = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
N_MID = [100, 200, 400, 800, 1600, 3200, 6400, 10000]
N_CHAIN = [5, 11, 21, 51, 101, 201, 501, 1001, 2001, 5001, 10001]

TRIALS = {  # (faster, fast)
    0.0: (4000, 1000),
    1.0: (4000, 4000),
}
TRIALS_MID = (2000, 400)


def configs():
    out = [(0.0, N) for N in N_PARALLEL]
    out += [(a, N) for a in ALPHAS_MID for N in N_MID]
    out += [(1.0, N) for N in N_CHAIN]
    return out


def genome_for(alpha, N):
    if alpha == 1.0:
        return chain(N)
    return interpolating(N, module_length(N, alpha))


def n_trials(engine, alpha):
    t = TRIALS.get(alpha, TRIALS_MID)
    return t[0] if engine == 'faster' else t[1]


def path(engine, alpha, N):
    return os.path.join(OUT, f'{engine}_a{alpha:.2f}_N{N}.npz')


def seed_base(engine, alpha, N):
    return (int(round(alpha * 100)) * 100_000 + N) * 10_000 + (0 if engine == 'faster' else 5_000)


# ── fast_circuit engine (one trial per task) ──────────────────────────────────

def fast_trial(args):
    """Mirror of run_strong_selection_fast(and_funct, NP, 0.001, x, inf, M_MAX)."""
    genome, seed = args
    from fast_circuit import FastCircuit
    from goals import and_funct
    random.seed(seed)
    np.random.seed(seed)
    mu = 0.001
    x = FastCircuit(genome)
    L = len(x.genome)
    F = x.fitness(and_funct)
    counts = np.zeros(5, dtype=np.int64)
    M = 0
    while F < 1.0 and M + 1 < M_MAX:
        np.random.exponential(1 / (NP * mu * L))      # keeps RNG stream identical
        xm = x.duplicate()
        xm.mutate()
        F2 = xm.fitness(and_funct)
        s = F2 - F
        if s != 0:
            if abs(-s * NP) < 250:
                pfix = (1 - np.exp(-s)) / (1 - np.exp(-s * NP))
            elif -s * NP > 0:
                pfix = 0.0
            else:
                pfix = 1 - np.exp(-s)
        else:
            pfix = 1 / NP
        counts[int(round(F * 4))] += 1
        M += 1
        if np.random.rand() < pfix:
            x = xm
            F = F2
    return M, counts


def _init_worker():
    from fast_circuit.circuit import warmup as w1
    from faster_circuit import warmup as w2
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        w1(); w2()


def faster_config(args):
    alpha, N = args
    from faster_circuit import run_strong_selection_faster
    from goals import and_funct
    t = time.perf_counter()
    M, counts = run_strong_selection_faster(
        and_funct, NP, genome_for(alpha, N), n_trials=n_trials('faster', alpha),
        seed=seed_base('faster', alpha, N), m_max=M_MAX)
    np.savez(path('faster', alpha, N), M=M, counts=counts,
             seconds=time.perf_counter() - t)
    return alpha, N, float(M.mean()), time.perf_counter() - t


def run_faster(pool):
    todo = [c for c in configs() if not os.path.exists(path('faster', *c))]
    for alpha, N, mean, dt in pool.imap_unordered(faster_config, todo):
        print(f'[faster] alpha={alpha:.2f} N={N:6d}  <M>={mean:10.1f}  {dt:7.1f}s', flush=True)


def run_fast(pool):
    todo = [c for c in configs() if not os.path.exists(path('fast', *c))]
    # cheapest first, so partial results are usable early
    todo.sort(key=lambda c: (c[1] ** 2 if c[0] < 1 else c[1]))
    for alpha, N in todo:
        g = genome_for(alpha, N)
        base = seed_base('fast', alpha, N)
        n = n_trials('fast', alpha)
        t = time.perf_counter()
        res = pool.map(fast_trial, [(g, base + j) for j in range(n)], chunksize=1)
        dt = time.perf_counter() - t
        M = np.array([r[0] for r in res], dtype=np.int64)
        counts = np.array([r[1] for r in res], dtype=np.int64)
        np.savez(path('fast', alpha, N), M=M, counts=counts, seconds=dt)
        print(f'[fast]   alpha={alpha:.2f} N={N:6d}  <M>={M.mean():10.1f}  {dt:7.1f}s', flush=True)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    os.makedirs(OUT, exist_ok=True)
    _init_worker()
    with Pool(workers, initializer=_init_worker) as pool:
        if which in ('faster', 'both'):
            run_faster(pool)
        if which in ('fast', 'both'):
            run_fast(pool)
