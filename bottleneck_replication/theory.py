"""
Exact results from the bottleneck note, plus a brute-force enumerator.

  exact_M_parallel(N, Np)   Eq. (3): <M> from the parallel start
  inv_p0_formula(N, L, Np)  Eq. (4): 1 / p0 for the interpolating start
  enumerate_p(genome, Np)   brute force over every (edge, source) pair with
                            the Methods weights; returns p_sigma and the
                            beneficial fraction
"""
import numpy as np
import numba

from faster_circuit.core import node_value, is_ancestor, pfix_table, POPCOUNT4

AND = 0b1000


def _ab(Np):
    pf = pfix_table(Np)
    return pf[4 + 2], pf[4 + 1], pf[4 + 3], pf[4 + 4]   # pi(1/2), pi(1/4), pi(3/4), pi(1)


def exact_M_parallel(N, Np=1e4):
    a, b, _, _ = _ab(Np)
    return ((2 * N + 1) * N ** 2 / (2 * (N - 1) * ((N - 1) * a + b))
            + ((2 * N + 1) * N - 2) / ((N - 1) * a))


def inv_p0_formula(N, L, Np=1e4):
    a, b, p34, p1 = _ab(Np)
    W0 = 2 * N + 1 - L * (L - 1) / N
    brace = (2 * ((N - L) * a + b)
             + 2 * b * ((L - 1) / 2 * (N - L) + (L - 1) * (L + 1) / 8)
             + 2 * (L - 1) * a
             + p34 * (L - 1) * (L - 3) / 4
             + N * (L - 1) / (2 * (N - 1)) * p1)
    return N * W0 / brace


@numba.njit(cache=True)
def _enumerate(genome, target, pfix):
    L = len(genome)
    n_gates = (L - 1) // 2
    n_nodes = n_gates + 2
    memo = np.zeros(n_nodes, dtype=np.int64)
    mstamp = np.zeros(n_nodes, dtype=np.int64)
    sstamp = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    epoch = 1
    out = genome[L - 1]
    m0 = POPCOUNT4[(~(node_value(genome, out, memo, mstamp, epoch, stack) ^ target)) & 0xF]

    w_valid = 0.0
    w_win = 0.0            # sum of weight * pfix over beneficial pairs
    w_ben = 0.0            # sum of weight over beneficial pairs
    for e in range(L):
        old = genome[e]
        if e == L - 1:
            w = 1.0 / (n_gates - 1)
            for node in range(2, n_nodes):
                if node == old:
                    continue
                genome[e] = node
                epoch += 1
                m = POPCOUNT4[(~(node_value(genome, node, memo, mstamp, epoch, stack) ^ target)) & 0xF]
                w_valid += w
                if m > m0:
                    w_ben += w
                    w_win += w * pfix[m - m0 + 4]
        else:
            g = e // 2
            w = 1.0 / n_gates
            for node in range(n_nodes):
                if node == old or node == g + 2:
                    continue
                if node >= 2:
                    epoch += 1
                    if is_ancestor(genome, node, g + 2, sstamp, epoch, stack):
                        continue
                genome[e] = node
                epoch += 1
                m = POPCOUNT4[(~(node_value(genome, out, memo, mstamp, epoch, stack) ^ target)) & 0xF]
                w_valid += w
                if m > m0:
                    w_ben += w
                    w_win += w * pfix[m - m0 + 4]
        genome[e] = old
    return w_win / w_valid, w_ben / w_valid


def enumerate_p(genome, Np=1e4, target=AND):
    """(p_sigma, beneficial fraction) of a genotype under the Methods weights."""
    g = np.asarray(genome, dtype=np.int64).copy()
    return _enumerate(g, target, pfix_table(Np))
