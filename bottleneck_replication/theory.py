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


# ── exact p_sigma in O(16 L) for a chain module in a pristine reservoir ───────

@numba.njit(cache=True)
def _nand(a, b):
    return (~(a & b)) & 0xF


@numba.njit(cache=True)
def _p_chain_reservoir(N, L, k, target, pfix):
    """
    Same quantity as enumerate_p, for the genotypes of starts.interpolating(N, L)
    (k = 0) and starts.bottleneck_state(N, L, k) (k odd >= 3), without looping
    over sources one by one.

    Idea: rewiring an input of chain gate j changes only v_j, and the output is
    then Phi_j(v_j), where Phi_j (a 16-entry table) composes the chain below j.
    The result depends on the new source only through its truth table, so it is
    enough to count the acyclic candidate sources (non-descendants of j) by
    truth-table value. Reservoir gates outside the module are neutral.
    """
    I1, I2, RES = 0b1100, 0b1010, 0b0111        # NAND(I1, I2) = 0b0111
    # chain values v[1..L]
    v = np.zeros(L + 1, dtype=np.int64)
    v[1] = _nand(I1, I2)
    for j in range(2, L + 1):
        v[j] = _nand(RES, v[j - 1]) if j == k else _nand(v[j - 1], v[j - 1])
    # Phi[j][t]: output if v_j is replaced by t
    phi = np.zeros((L + 1, 16), dtype=np.int64)
    for t in range(16):
        phi[L, t] = t
    for j in range(L, 1, -1):
        for t in range(16):
            g = _nand(RES, t) if j == k else _nand(t, t)
            phi[j - 1, t] = phi[j, g]

    def gain(out, m0):
        return POPCOUNT4[(~(out ^ target)) & 0xF] - m0

    m0 = POPCOUNT4[(~(v[L] ^ target)) & 0xF]
    n_res = N - L
    w_valid = 0.0
    w_win = 0.0
    w_ben = 0.0
    wg = 1.0 / N
    prefix = np.zeros(16, dtype=np.int64)       # value counts of chain gates 1..j-1
    cnt = np.zeros(16, dtype=np.int64)
    for j in range(1, L + 1):
        # the two input edges of chain gate j: (current source value, other input value)
        for slot in range(2):
            if j == 1:
                cur = I1 if slot == 0 else I2
                w = I2 if slot == 0 else I1
            elif j == k:
                cur = RES if slot == 0 else v[j - 1]       # slot 0 is the recruited copy x
                w = v[j - 1] if slot == 0 else RES
            else:
                cur = v[j - 1]
                w = v[j - 1]
            for t in range(16):
                cnt[t] = prefix[t]
            cnt[I1] += 1
            cnt[I2] += 1
            cnt[RES] += n_res
            cnt[cur] -= 1
            for t in range(16):
                c = cnt[t]
                if c == 0:
                    continue
                w_valid += c * wg
                d = gain(phi[j, _nand(t, w)], m0)
                if d > 0:
                    w_ben += c * wg
                    w_win += c * wg * pfix[d + 4]
        prefix[v[j]] += 1
    # output edge: any gate other than chain gate L
    wo = 1.0 / (N - 1)
    for t in range(16):
        cnt[t] = 0
    for j in range(1, L):
        cnt[v[j]] += 1
    cnt[RES] += n_res
    for t in range(16):
        c = cnt[t]
        if c == 0:
            continue
        w_valid += c * wo
        d = gain(t, m0)
        if d > 0:
            w_ben += c * wo
            w_win += c * wo * pfix[d + 4]
    # reservoir gates not feeding the module: 2 edges x N acyclic sources, neutral
    n_free = n_res - (1 if k > 0 else 0)
    w_valid += 2 * n_free * N * wg
    # the recruited copy x (k > 0): its value feeds chain gate k
    if k > 0:
        for slot in range(2):
            cur = I1 if slot == 0 else I2
            w = I2 if slot == 0 else I1
            for t in range(16):
                cnt[t] = 0
            for j in range(1, k):
                cnt[v[j]] += 1
            cnt[I1] += 1
            cnt[I2] += 1
            cnt[RES] += n_res - 1
            cnt[cur] -= 1
            for t in range(16):
                c = cnt[t]
                if c == 0:
                    continue
                w_valid += c * wg
                d = gain(phi[k, _nand(_nand(t, w), v[k - 1])], m0)
                if d > 0:
                    w_ben += c * wg
                    w_win += c * wg * pfix[d + 4]
    return w_win / w_valid, w_ben / w_valid


def p_chain_reservoir(N, L, k=0, Np=1e4, target=AND):
    """Exact (p_sigma, beneficial fraction) for interpolating(N, L) or bottleneck_state(N, L, k)."""
    return _p_chain_reservoir(N, L, k, target, pfix_table(Np))


def bottleneck_k(L):
    """Odd chain gate near the middle of the module, as in the note's Table 4."""
    k = L // 2
    return max(3, k if k % 2 else k + 1)
