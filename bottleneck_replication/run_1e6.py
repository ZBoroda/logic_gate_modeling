"""
faster_circuit simulations of the interpolating family up to N = 10^6.

Same model as run_sims.py (N_P = 1e4, target AND, Methods mutation distribution).
Results: bottleneck_replication/results_1e6/a<alpha>_N<N>.npz with M, counts, seconds.
Each config is split into chunks run on separate workers; trial j of a config
is seeded with seed_base + j, so the result does not depend on the chunking.
Existing files are skipped.

Usage:  python -m bottleneck_replication.run_1e6 [n_workers]
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from bottleneck_replication.starts import interpolating, chain, module_length

NP = 1e4
M_MAX = 10**9
OUT = os.path.join(os.path.dirname(__file__), 'results_1e6')

ALPHAS_MID = [0.2, 0.4, 0.5, 0.6, 0.75, 0.9]
N_GRID = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000,
          100_000, 200_000, 500_000, 1_000_000]
N_PARALLEL = [5, 10, 20] + N_GRID
N_CHAIN = [5, 11, 21, 51, 101, 201, 501, 1001, 2001, 5001, 10_001, 20_001, 50_001,
           100_001, 200_001, 500_001, 1_000_001]
TRIALS = {0.0: 1000, 1.0: 4000}
TRIALS_MID = 1000
CHUNK = 50


def configs():
    out = [(0.0, N) for N in N_PARALLEL]
    out += [(a, N) for a in ALPHAS_MID for N in N_GRID]
    out += [(1.0, N) for N in N_CHAIN]
    return out


def genome_for(alpha, N):
    if alpha == 1.0:
        return chain(N)
    return interpolating(N, module_length(N, alpha))


def path(alpha, N):
    return os.path.join(OUT, f'a{alpha:.2f}_N{N}.npz')


def seed_base(alpha, N):
    return configs().index((alpha, N)) * 100_000          # < 2**32, distinct per trial


def run_chunk(args):
    alpha, N, start, n = args
    from faster_circuit import run_strong_selection_faster
    from goals import and_funct
    t = time.perf_counter()
    M, counts = run_strong_selection_faster(and_funct, NP, genome_for(alpha, N), n_trials=n,
                                            seed=seed_base(alpha, N) + start, m_max=M_MAX)
    return alpha, N, start, M, counts, time.perf_counter() - t


def _init():
    from faster_circuit import warmup
    warmup()


if __name__ == '__main__':
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    os.makedirs(OUT, exist_ok=True)
    todo = [c for c in configs() if not os.path.exists(path(*c))]
    tasks = []
    for alpha, N in todo:
        n = TRIALS.get(alpha, TRIALS_MID)
        tasks += [(alpha, N, s, min(CHUNK, n - s)) for s in range(0, n, CHUNK)]
    tasks.sort(key=lambda t: t[1])                     # small N first
    parts = {}
    _init()
    with Pool(workers, initializer=_init) as pool:
        for alpha, N, start, M, counts, dt in pool.imap_unordered(run_chunk, tasks):
            parts.setdefault((alpha, N), []).append((start, M, counts, dt))
            n = TRIALS.get(alpha, TRIALS_MID)
            got = sum(len(p[1]) for p in parts[(alpha, N)])
            if got == n:
                ps = sorted(parts.pop((alpha, N)), key=lambda p: p[0])
                M = np.concatenate([p[1] for p in ps])
                counts = np.concatenate([p[2] for p in ps])
                cpu = sum(p[3] for p in ps)
                np.savez(path(alpha, N), M=M, counts=counts, seconds=cpu)
                print(f'alpha={alpha:.2f} N={N:8d} <M>={M.mean():12.1f} max={M.max():10d} '
                      f'cpu={cpu:8.1f}s', flush=True)
