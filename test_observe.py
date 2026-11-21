"""
Tests for faster_circuit.observe:
  1. instrumented runs are bit-identical to faster_circuit.run_batch (same M, counts)
  2. stats are self-consistent (fix + lost = M, residents = fixations + 1)
  3. exact p_b_given_c and computing_edges match brute-force enumeration
     (every computing edge, every alternative source, explicit cycle test)
  4. sampled-edge f_b is unbiased
"""
import random

import numpy as np

from faster_circuit.core import run_batch, pfix_table, target_bits, fitness
from faster_circuit.observe import (run_batch_obs, p_b_given_c, computing_edges,
                                    STAT_COLS, warmup)
from fast_circuit.circuit import FastCircuit
from goals import and_funct, or_funct
from bottleneck_replication.starts import parallel, chain, interpolating

C = {k: i for i, k in enumerate(STAT_COLS)}


def random_dag(N, rng):
    g = []
    for j in range(N):
        g += [rng.randrange(j + 2), rng.randrange(j + 2)]       # sources earlier: acyclic
    g.append(rng.randrange(N) + 2)
    # relabel gates randomly to avoid topological ordering by index
    perm = list(range(2, N + 2)); rng.shuffle(perm)
    mp = {0: 0, 1: 1, **{i + 2: perm[i] for i in range(N)}}
    out = [0] * (2 * N + 1)
    for j in range(N):
        nj = mp[j + 2] - 2
        out[2 * nj], out[2 * nj + 1] = mp[g[2 * j]], mp[g[2 * j + 1]]
    out[-1] = mp[g[-1]]
    return out


def brute(genome, f):
    """(computing edges, P(b|c)) straight from the Methods definitions."""
    N = (len(genome) - 1) // 2
    out = genome[-1]
    mod, stack = set(), [out]
    while stack:
        v = stack.pop()
        if v < 2 or v in mod:
            continue
        mod.add(v)
        stack += [genome[2 * (v - 2)], genome[2 * (v - 2) + 1]]
    edges = [2 * (v - 2) + s for v in mod for s in (0, 1)] + [2 * N]
    f0 = fitness(genome, f)
    fbs = []
    for e in edges:
        cur = genome[e]
        if e == 2 * N:
            alts = [u for u in range(2, N + 2) if u != cur]
        else:
            v = e // 2 + 2
            alts = [u for u in range(N + 2) if u not in (cur, v)]
        ben = tot = 0
        for u in alts:
            g = list(genome); g[e] = u
            if FastCircuit(g).has_cycle():
                continue
            tot += 1
            ben += fitness(g, f) > f0
        fbs.append(ben / tot)
    return len(edges), float(np.mean(fbs))


def test_identical_to_run_batch():
    for gen, Np in [(parallel(30), 1e4), (chain(31), 1e4), (interpolating(200, 15), 1e4),
                    (parallel(40), 10.0)]:
        g = np.array(gen, dtype=np.int64)
        M, cnt = run_batch(g, 0b1000, pfix_table(Np), 10**7, 7, 40)
        st, cnt2 = run_batch_obs(g, 0b1000, pfix_table(Np), 10**7, 7, 40, 0.5, 5)
        assert (st[:, 0] == M).all() and (cnt == cnt2).all()
        fix = st[:, 1:4].sum(1); lost = st[:, 4:7].sum(1)
        assert (fix + lost == M).all()
        assert (st[:, C['n_res']] == fix + 1).all()
    print('PASS identical to run_batch; stats consistent')


def test_p_b_given_c():
    rng = random.Random(3)
    n = 0
    for _ in range(150):
        N = rng.randrange(2, 12)
        g = random_dag(N, rng)
        for f in (and_funct, or_funct):
            ce, fb = brute(g, f)
            assert computing_edges(g) == ce
            assert abs(p_b_given_c(g, f) - fb) < 1e-12, (g, fb, p_b_given_c(g, f))
            n += 1
    for g in (parallel(9), chain(9), interpolating(15, 5)):
        ce, fb = brute(g, and_funct)
        assert computing_edges(g) == ce and abs(p_b_given_c(g, and_funct) - fb) < 1e-12
    print(f'PASS p_b_given_c / computing_edges vs brute force ({n} random DAGs)')


def test_sampled_unbiased():
    g = chain(41)
    exact = p_b_given_c(g, and_funct)
    est = np.mean([p_b_given_c(g, and_funct, k_edges=5, seed=s) for s in range(4000)])
    assert abs(est - exact) < 0.01, (est, exact)
    print(f'PASS sampled f_b unbiased: {est:.4f} vs exact {exact:.4f}')


if __name__ == '__main__':
    warmup()
    test_identical_to_run_batch()
    test_p_b_given_c()
    test_sampled_unbiased()
