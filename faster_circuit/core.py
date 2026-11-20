"""
O(module-size) SSWM Moran simulator for NAND networks, Numba-compiled.

Implements the algorithm of the note "Simulating the evolution of Boolean
networks quickly" (see faster_circuit/README.md):

  (i)   truth tables as 4-bit integers, all four input rows in one op
  (ii)  lazy evaluation backwards from the output (only the computing module)
  (iii) local cycle test: search only the ancestors of the proposed new source
  (iv)  O(1) source sampling by index arithmetic
  (v)   mutate in place, revert if the mutant is lost
  (vi)  no per-step time draws; T | M ~ Gamma(M, tau / Lambda) after the run

Genome format is the same as Circuit / FastCircuit:
  [in1_g0, in2_g0, ..., in1_g(N-1), in2_g(N-1), out]
Nodes 0, 1 are the inputs I1, I2; gates are nodes 2 .. N+1.

Truth-table bit i holds input row i, rows ordered (I1, I2) = 00, 01, 10, 11:
  I1 -> 0b1100, I2 -> 0b1010, AND -> 0b1000, NAND(a, b) = ~(a & b) & 0xF.

Mutation distribution (the paper's Methods): pick one of the 2N + 1 edges
uniformly; an edge into gate g is redirected to a uniform node other than
g and its current source (N candidates); the output edge to a uniform gate
other than the current one (N - 1 candidates). Cyclic proposals are redrawn
and not counted.
"""
import math

import numpy as np
import numba

NUM_INPUTS = 2
INPUT_TT = np.array([0b1100, 0b1010], dtype=np.int64)
POPCOUNT4 = np.array([bin(i).count('1') for i in range(16)], dtype=np.int64)


def target_bits(f) -> int:
    """Convert a goals.py-style function f([i1, i2]) -> bool to a 4-bit table."""
    t = 0
    for i in range(4):
        if f([bool(i & 2), bool(i & 1)]):
            t |= 1 << i
    return t


def pfix_table(Np: float) -> np.ndarray:
    """
    Moran fixation probability pi(s) for s = d/4, d = -4..4, indexed by d + 4.
    Same overflow guard as fast_circuit.models.run_strong_selection_fast.
    """
    tab = np.empty(9, dtype=np.float64)
    for d in range(-4, 5):
        s = d / 4.0
        if d == 0:
            p = 1.0 / Np
        elif abs(s * Np) < 250:
            p = (1 - math.exp(-s)) / (1 - math.exp(-s * Np))
        elif s < 0:
            p = 0.0
        else:
            p = 1 - math.exp(-s)
        tab[d + 4] = p
    return tab


# ── core compiled routines ────────────────────────────────────────────────────

@numba.njit(cache=True)
def node_value(genome, node, memo, stamp, epoch, stack):
    """
    4-bit truth table of `node`, evaluated lazily backwards over its ancestors.
    Iterative post-order with memoisation; `stamp[n] == epoch` marks memo[n]
    valid, so no O(N) clearing is ever needed. Visits only ancestors of node.
    """
    if node < 2:
        return INPUT_TT[node]
    sp = 0
    stack[sp] = node
    sp += 1
    while sp > 0:
        n = stack[sp - 1]
        if stamp[n] == epoch:
            sp -= 1
            continue
        g = n - 2
        a = genome[2 * g]
        b = genome[2 * g + 1]
        ready = True
        if a >= 2 and stamp[a] != epoch:
            stack[sp] = a
            sp += 1
            ready = False
        if b >= 2 and stamp[b] != epoch:
            stack[sp] = b
            sp += 1
            ready = False
        if ready:
            va = INPUT_TT[a] if a < 2 else memo[a]
            vb = INPUT_TT[b] if b < 2 else memo[b]
            memo[n] = (~(va & vb)) & 0xF
            stamp[n] = epoch
            sp -= 1
    return memo[node]


@numba.njit(cache=True)
def is_ancestor(genome, start, target, stamp, epoch, stack):
    """True iff `target` is `start` or an ancestor of it (DFS over parents)."""
    sp = 0
    stack[sp] = start
    sp += 1
    while sp > 0:
        sp -= 1
        n = stack[sp]
        if n == target:
            return True
        if n < 2 or stamp[n] == epoch:
            continue
        stamp[n] = epoch
        g = n - 2
        stack[sp] = genome[2 * g]
        sp += 1
        stack[sp] = genome[2 * g + 1]
        sp += 1
    return False


@numba.njit(cache=True)
def matches(genome, target, memo, stamp, epoch, stack):
    """Number of input rows (0..4) on which the output agrees with target."""
    out = genome[len(genome) - 1]
    v = node_value(genome, out, memo, stamp, epoch, stack)
    return POPCOUNT4[(~(v ^ target)) & 0xF]


@numba.njit(cache=True)
def _run_one(genome, target, pfix, m_max, counts):
    """
    One SSWM run, in place on `genome`. Returns the number of mutations M.
    counts[k] += mutations made while the resident had k matching rows.
    Stops when all 4 rows match, or after m_max mutations (then M = m_max).
    """
    L = len(genome)
    n_gates = (L - 1) // 2
    n_nodes = n_gates + 2
    memo = np.zeros(n_nodes, dtype=np.int64)
    mstamp = np.zeros(n_nodes, dtype=np.int64)
    sstamp = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    epoch = 1

    m = matches(genome, target, memo, mstamp, epoch, stack)
    M = 0
    while m < 4 and M < m_max:
        e = np.random.randint(0, L)
        if e == L - 1:                                   # output edge
            cur = genome[e] - 2
            k = np.random.randint(0, n_gates - 1)
            new = (k if k < cur else k + 1) + 2
            old = genome[e]
            genome[e] = new
        else:                                            # input edge of gate g
            g = e // 2
            old = genome[e]
            while True:                                  # O(1) uniform source
                i = np.random.randint(0, n_gates + 1)
                if i < 2:
                    new = i
                elif i - 2 < g:
                    new = i
                else:
                    new = i + 1
                if new != old:
                    break
            if new >= 2:                                 # local cycle test
                epoch += 1
                if is_ancestor(genome, new, g + 2, sstamp, epoch, stack):
                    continue                             # cyclic: redraw, not counted
            genome[e] = new                              # mutate in place

        epoch += 1
        m2 = matches(genome, target, memo, mstamp, epoch, stack)
        counts[m] += 1
        M += 1
        if np.random.random() < pfix[m2 - m + 4]:
            m = m2                                       # fixes
        else:
            genome[e] = old                              # lost: revert
    return M


@numba.njit(cache=True)
def run_batch(genome0, target, pfix, m_max, seed, n_trials):
    """
    n_trials independent runs from genome0. Trial j is seeded with seed + j.
    Returns M[n_trials] and counts[n_trials, 5] (mutations spent at each
    number of matching rows; counts[:, 0] is the first step from f = 0).
    """
    Ms = np.empty(n_trials, dtype=np.int64)
    counts = np.zeros((n_trials, 5), dtype=np.int64)
    for j in range(n_trials):
        np.random.seed(seed + j)
        g = genome0.copy()
        Ms[j] = _run_one(g, target, pfix, m_max, counts[j])
    return Ms, counts


# ── Python-facing helpers ─────────────────────────────────────────────────────

def fitness(genome, f) -> float:
    """Fitness of a genome against target function f (for testing)."""
    genome = np.asarray(genome, dtype=np.int64)
    n_nodes = (len(genome) - 1) // 2 + 2
    memo = np.zeros(n_nodes, dtype=np.int64)
    stamp = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    return matches(genome, target_bits(f), memo, stamp, 1, stack) / 4.0


def has_cycle_via(genome, gate_node, new_source) -> bool:
    """Would setting an input of gate_node to new_source close a cycle?"""
    genome = np.asarray(genome, dtype=np.int64)
    n_nodes = (len(genome) - 1) // 2 + 2
    stamp = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    if new_source < 2:
        return False
    return bool(is_ancestor(genome, new_source, gate_node, stamp, 1, stack))


def run_strong_selection_faster(f, Np: float, genome, n_trials: int = 1,
                                seed: int = 0, m_max: int = 10**9):
    """
    Run n_trials SSWM Moran runs to target f from `genome`.

    Returns (M, counts): M[j] is the number of mutations of trial j,
    counts[j, k] the number spent with k of 4 rows correct.
    Physical time, if wanted: T_j ~ Gamma(M_j, scale = tau / (Np * mu * (2N+1))).
    """
    g0 = np.asarray(genome, dtype=np.int64)
    return run_batch(g0, target_bits(f), pfix_table(Np), m_max, seed, n_trials)


def warmup():
    g = np.array([0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    run_batch(g, 0b1000, pfix_table(1e4), 1000, 0, 2)
