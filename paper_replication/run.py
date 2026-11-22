"""
Simulations behind every data figure of the paper (arXiv 2610.40635), with
faster_circuit.observe, extended to large N.

  exp        start     target       N_P    used by
  par_1e4    parallel  AND          1e4    Fig 3, 4, S2, S6A
  chain_1e4  chain     AND          1e4    Fig 5, S6B
  par_10     parallel  AND          10     Fig S4 (and S3 / S4B comparison)
  chain_10   chain     AND          10     Fig S5
  andnot_10  parallel  I1 AND ~I2   10     Fig S3A
  or_10      parallel  OR           10     Fig S3B
  true_10    parallel  TRUE         10     Fig S3C
  rw         parallel  AND          (all mutations accepted)   Fig S1
  andnot_1e4, or_1e4, true_1e4   as the _10 runs at N_P = 1e4   Fig S3

Results: paper_replication/results/<exp>_N<N>.npz with stats (columns
faster_circuit.observe.STAT_COLS), counts, seconds, p_res, k_edges.
Trial j of a config is seeded with seed_base + j, so chunking does not change
results. Existing files are skipped; the most expensive configs run first.
Each file records fb_def, the f_b convention it was measured with.

Usage:  python -m paper_replication.run [n_workers] [exp ...]
"""
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

from bottleneck_replication.starts import parallel, chain

OUT = os.path.join(os.path.dirname(__file__), 'results')
M_MAX = 10**9
R_FB = 30          # f_b measured on ~R_FB residents per run (all, if fewer)
K_EDGES = 64       # f_b exact over all computing edges up to this many, else sampled

GRID = [5, 10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10_000, 20_000, 40_000,
        100_000, 200_000, 400_000, 1_000_000]
GRID_ODD = [n + 1 if n % 2 == 0 else n for n in GRID]

AND, ANDNOT, OR, TRUE = 0b1000, 0b0100, 0b1110, 0b1111

# name: (start, target, Np (None = random walk), N list, trials, measure f_b)
EXPS = {
    'par_1e4':   ('parallel', AND, 1e4, GRID, 1000, True),
    'chain_1e4': ('chain', AND, 1e4, GRID_ODD, 1000, True),
    'par_10':    ('parallel', AND, 10.0, GRID[:14], 500, True),
    'chain_10':  ('chain', AND, 10.0, GRID_ODD, 500, True),
    'andnot_10': ('parallel', ANDNOT, 10.0, GRID[:11], 500, False),
    'or_10':     ('parallel', OR, 10.0, GRID[:11], 500, False),
    'true_10':   ('parallel', TRUE, 10.0, GRID[:11], 500, False),
    'rw':        ('parallel', AND, None, GRID[:14], 500, False),
    'andnot_1e4': ('parallel', ANDNOT, 1e4, GRID[:14], 500, False),
    'or_1e4':    ('parallel', OR, 1e4, GRID[:14], 500, False),
    'true_1e4':  ('parallel', TRUE, 1e4, GRID[:14], 500, False),
}
CHUNK = 25
FB_DEF = 'edge_pooled_no_final'   # faster_circuit.observe f_b convention (see its docstring)
# rough relative cost per run, for scheduling the most expensive chunks first
COST = {'par_10': 40.0, 'rw': 40.0, 'andnot_10': 20.0, 'or_10': 10.0, 'true_10': 5.0,
        'andnot_1e4': 30.0, 'or_1e4': 10.0, 'true_1e4': 10.0}


def path(exp, N):
    return os.path.join(OUT, f'{exp}_N{N}.npz')


def seed_base(exp, N):
    return (zlib.crc32(exp.encode()) % 20_000 + N % 9973 * 20_000) * 10 % (2**31 - 10**6)


def run_chunk(args):
    exp, N, start, n = args
    from faster_circuit.observe import run_observed
    kind, tgt, Np, _, _, fb = EXPS[exp]
    genome = parallel(N) if kind == 'parallel' else chain(N)
    t = time.perf_counter()
    st, counts = run_observed(tgt, Np or 1.0, genome, n_trials=n, seed=seed_base(exp, N) + start,
                              m_max=M_MAX, p_res=-R_FB if fb else 0.0, k_edges=K_EDGES,
                              random_walk=Np is None)
    return exp, N, start, st, counts, time.perf_counter() - t


def _init():
    from faster_circuit.observe import warmup
    warmup()


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    exps = sys.argv[2:] or list(EXPS)
    os.makedirs(OUT, exist_ok=True)
    tasks = []
    for exp in exps:
        _, _, _, Ns, trials, _ = EXPS[exp]
        for N in Ns:
            if os.path.exists(path(exp, N)):
                continue
            tasks += [(exp, N, s, min(CHUNK, trials - s)) for s in range(0, trials, CHUNK)]
    # most expensive first: heavy-tailed run times, so the big chunks must not
    # start last and leave one core working alone for hours
    tasks.sort(key=lambda t: -t[1] * COST.get(t[0], 1.0))
    print(f'{len(tasks)} chunks', flush=True)
    parts = {}
    _init()
    with Pool(workers, initializer=_init) as pool:
        for exp, N, start, st, counts, dt in pool.imap_unordered(run_chunk, tasks):
            parts.setdefault((exp, N), []).append((start, st, counts, dt))
            trials = EXPS[exp][4]
            if sum(len(p[1]) for p in parts[(exp, N)]) == trials:
                ps = sorted(parts.pop((exp, N)), key=lambda p: p[0])
                st = np.concatenate([p[1] for p in ps])
                counts = np.concatenate([p[2] for p in ps])
                cpu = sum(p[3] for p in ps)
                np.savez(path(exp, N), stats=st, counts=counts, seconds=cpu,
                         r_fb=R_FB, k_edges=K_EDGES, fb_def=FB_DEF)
                print(f'{time.strftime("%H:%M:%S")} {exp:10s} N={N:8d} <M>={st[:, 0].mean():12.1f} '
                      f'ce={st[:, 8].sum() / st[:, 7].sum():9.1f} '
                      f'fb={np.nanmean(st[:, 9] / st[:, 10]) if st[:, 10].any() else float("nan"):.3f} '
                      f'capped={int(st[:, 11].sum())} cpu={cpu:8.1f}s', flush=True)


if __name__ == '__main__':
    main()
