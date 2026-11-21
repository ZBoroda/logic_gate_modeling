"""
Instrumented faster_circuit runs: the observables of the paper's figures.

Per run this records, besides M:
  * fixating / non-fixating mutations, split by type (beneficial, deleterious,
    neutral)                                               Fig. 3C-D, S4B, S5B
  * computing edges of every resident network (initial + each fixed mutant),
    averaged over residents                                Fig. 4B, 5B; P(c)
  * f_b, the fraction of beneficial alternatives per computing edge, averaged
    over the computing edges of a resident and then over residents   Fig. 4C,
    5C, S4D, S5D; P(b|c)

Definitions follow the paper's Methods 6.3-6.6:
  computing edge  an edge on a path from the inputs to the output O. With a
                  DAG every ancestor of O reaches the inputs, so the computing
                  edges are the 2 input edges of each module gate plus the output
                  edge: n_c = 2 |module| + 1.
  f_b(e)          for computing edge u->v, the alternatives are every node except
                  u, v and O (every gate except u when v = O); those closing a
                  cycle are not allowed. f_b(e) = #beneficial / #allowed.

f_b(e) is computed exactly without trying sources one by one. Rewiring an input
of gate v changes only v's truth table, to NAND(t, w), where t is the new
source's table and w that of v's other input. The output is then Phi_v(NAND(t,
w)), so only the histogram of truth tables over allowed sources is needed:
(all nodes) - (descendants of v) - v - u. Cost per edge O(|desc(v)| + 16
|module|), per resident O(N) for the histogram and the forward adjacency.

Cost control (large N): a resident's f_b is measured with probability p_res
(uniform thinning, so the per-run mean over measured residents is an unbiased
estimate of the mean over all residents), and when it has more than k_edges
computing edges, k_edges of them are sampled uniformly with replacement. With
p_res = 1 and k_edges >= n_c the measurement is exact. Thinning and edge
sampling use a private splitmix64 stream, so the simulation itself (M, counts,
every fixation) is bit-identical to faster_circuit.run_batch with the same seed.
"""
import numpy as np
import numba

from faster_circuit.core import (INPUT_TT, POPCOUNT4, node_value, is_ancestor,
                                 mark_module, pfix_table, target_bits)

# columns of the per-run stats array
STAT_COLS = ['M', 'fix_ben', 'fix_del', 'fix_neu', 'lost_ben', 'lost_del', 'lost_neu',
             'n_res', 'ce_sum', 'fb_sum', 'fb_n', 'capped']
S = {k: i for i, k in enumerate(STAT_COLS)}


# ── private RNG (does not touch numpy's stream) ───────────────────────────────

@numba.njit(cache=True)
def _sm_next(state):
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@numba.njit(cache=True)
def _sm_uniform(state):
    return (_sm_next(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@numba.njit(cache=True)
def _sm_int(state, n):
    return np.int64(_sm_next(state) % np.uint64(n))


# ── exact f_b of one resident ─────────────────────────────────────────────────

@numba.njit(cache=True)
def _all_tables(genome, tt, stamp, epoch, stack):
    """Truth table of every node into tt (shares one memo epoch). O(N)."""
    n_nodes = (len(genome) - 1) // 2 + 2
    tt[0] = INPUT_TT[0]
    tt[1] = INPUT_TT[1]
    for n in range(2, n_nodes):
        node_value(genome, n, tt, stamp, epoch, stack)


@numba.njit(cache=True)
def _forward_csr(genome, ptr, adj, fill):
    """Children lists: adj[ptr[u]:ptr[u+1]] are the gates reading node u."""
    n_gates = (len(genome) - 1) // 2
    n_nodes = n_gates + 2
    for i in range(n_nodes + 1):
        ptr[i] = 0
    for e in range(2 * n_gates):
        ptr[genome[e] + 1] += 1
    for i in range(n_nodes):
        ptr[i + 1] += ptr[i]
    for i in range(n_nodes):
        fill[i] = ptr[i]
    for e in range(2 * n_gates):
        u = genome[e]
        adj[fill[u]] = e // 2 + 2
        fill[u] += 1


@numba.njit(cache=True)
def _edge_fb(genome, e, target, m0, tt, ptr, adj, hist_all,
             dstamp, depoch, memo2, stamp2, epoch2, stack, queue):
    """
    Exact f_b of computing edge e. Returns (f_b, new epoch2, new depoch).
    hist_all[t] = number of nodes (inputs + gates) with truth table t.
    """
    L = len(genome)
    out = genome[L - 1]
    cnt = np.zeros(16, dtype=np.int64)
    if e == L - 1:
        # alternatives: every gate except the current one; never cyclic
        for t in range(16):
            cnt[t] = hist_all[t]
        cnt[INPUT_TT[0]] -= 1
        cnt[INPUT_TT[1]] -= 1
        cnt[tt[out]] -= 1
        ben = 0
        tot = 0
        for t in range(16):
            if cnt[t] > 0:
                tot += cnt[t]
                if POPCOUNT4[(~(t ^ target)) & 0xF] > m0:
                    ben += cnt[t]
        return ben / tot, epoch2, depoch

    v = e // 2 + 2
    cur = genome[e]
    w = tt[genome[e ^ 1]]
    # descendants of v (excluding v): forward BFS, histogram of their tables
    depoch += 1
    for t in range(16):
        cnt[t] = hist_all[t]
    dstamp[v] = depoch
    qh = 0
    qt = 0
    queue[qt] = v
    qt += 1
    while qh < qt:
        u = queue[qh]
        qh += 1
        for j in range(ptr[u], ptr[u + 1]):
            c = adj[j]
            if dstamp[c] != depoch:
                dstamp[c] = depoch
                cnt[tt[c]] -= 1
                queue[qt] = c
                qt += 1
    cnt[tt[v]] -= 1
    cnt[tt[cur]] -= 1
    # Phi_v(x): output when v's table is forced to x (only the x values needed)
    phi = np.full(16, -1, dtype=np.int64)
    ben = 0
    tot = 0
    for t in range(16):
        c = cnt[t]
        if c <= 0:
            continue
        tot += c
        x = (~(t & w)) & 0xF
        if phi[x] < 0:
            epoch2 += 1
            memo2[v] = x
            stamp2[v] = epoch2
            phi[x] = node_value(genome, out, memo2, stamp2, epoch2, stack)
        if POPCOUNT4[(~(phi[x] ^ target)) & 0xF] > m0:
            ben += c
    return ben / tot, epoch2, depoch


@numba.njit(cache=True)
def _resident_fb(genome, target, m0, mod_list, n_mod, k_edges, rng,
                 tt, tstamp, tepoch, ptr, adj, fill, dstamp, depoch,
                 memo2, stamp2, epoch2, stack, queue):
    """Mean f_b over the computing edges of the current genome (exact if
    k_edges >= 2 n_mod + 1, else k_edges sampled edges). Returns
    (fb, tepoch, depoch, epoch2)."""
    L = len(genome)
    tepoch += 1
    _all_tables(genome, tt, tstamp, tepoch, stack)
    _forward_csr(genome, ptr, adj, fill)
    n_nodes = (L - 1) // 2 + 2
    hist_all = np.zeros(16, dtype=np.int64)
    for n in range(n_nodes):
        hist_all[tt[n]] += 1
    n_ce = 2 * n_mod + 1
    s = 0.0
    if n_ce <= k_edges:
        for i in range(n_ce):
            e = L - 1 if i == n_ce - 1 else 2 * (mod_list[i // 2] - 2) + (i & 1)
            f, epoch2, depoch = _edge_fb(genome, e, target, m0, tt, ptr, adj, hist_all,
                                         dstamp, depoch, memo2, stamp2, epoch2, stack, queue)
            s += f
        return s / n_ce, tepoch, depoch, epoch2
    for _ in range(k_edges):
        i = _sm_int(rng, n_ce)
        e = L - 1 if i == n_ce - 1 else 2 * (mod_list[i // 2] - 2) + (i & 1)
        f, epoch2, depoch = _edge_fb(genome, e, target, m0, tt, ptr, adj, hist_all,
                                     dstamp, depoch, memo2, stamp2, epoch2, stack, queue)
        s += f
    return s / k_edges, tepoch, depoch, epoch2


# ── instrumented run (same random draws as core._run_one) ─────────────────────

@numba.njit(cache=True)
def _run_one_obs(genome, target, pfix, m_max, counts, st, p_res, k_edges, rng):
    L = len(genome)
    n_gates = (L - 1) // 2
    n_nodes = n_gates + 2
    memo = np.zeros(n_nodes, dtype=np.int64)
    mstamp = np.zeros(n_nodes, dtype=np.int64)
    sstamp = np.zeros(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    in_mod = np.zeros(n_nodes, dtype=np.bool_)
    mod_list = np.empty(n_nodes, dtype=np.int64)
    epoch = 1
    do_fb = p_res > 0
    nb = n_nodes if do_fb else 1
    tt = np.zeros(nb, dtype=np.int64)
    tstamp = np.zeros(nb, dtype=np.int64)
    ptr = np.zeros(nb + 1, dtype=np.int64)
    adj = np.zeros(2 * nb, dtype=np.int64)
    fill = np.zeros(nb, dtype=np.int64)
    dstamp = np.zeros(nb, dtype=np.int64)
    memo2 = np.zeros(nb, dtype=np.int64)
    stamp2 = np.zeros(nb, dtype=np.int64)
    queue = np.empty(nb, dtype=np.int64)
    tepoch = 0
    depoch = 0
    epoch2 = 0

    m = matches_(genome, target, memo, mstamp, epoch, stack)
    n_mod = mark_module(genome, in_mod, mod_list, 0, stack)
    # resident 0
    st[7] += 1
    st[8] += 2 * n_mod + 1
    if do_fb and _sm_uniform(rng) < p_res:
        fb, tepoch, depoch, epoch2 = _resident_fb(genome, target, m, mod_list, n_mod, k_edges, rng,
                                                  tt, tstamp, tepoch, ptr, adj, fill, dstamp, depoch,
                                                  memo2, stamp2, epoch2, stack, queue)
        st[9] += fb
        st[10] += 1
    M = 0
    while m < 4 and M < m_max:
        e = np.random.randint(0, L)
        if e == L - 1:
            cur = genome[e] - 2
            k = np.random.randint(0, n_gates - 1)
            new = (k if k < cur else k + 1) + 2
            old = genome[e]
            genome[e] = new
        else:
            g = e // 2
            old = genome[e]
            while True:
                i = np.random.randint(0, n_gates + 1)
                if i < 2:
                    new = i
                elif i - 2 < g:
                    new = i
                else:
                    new = i + 1
                if new != old:
                    break
            if new >= 2:
                epoch += 1
                if is_ancestor(genome, new, g + 2, sstamp, epoch, stack):
                    continue
            genome[e] = new

        touches_module = e == L - 1 or in_mod[e // 2 + 2]
        if touches_module:
            epoch += 1
            m2 = matches_(genome, target, memo, mstamp, epoch, stack)
        else:
            m2 = m
        counts[m] += 1
        M += 1
        kind = 0 if m2 > m else (1 if m2 < m else 2)          # ben, del, neu
        if np.random.random() < pfix[m2 - m + 4]:
            st[1 + kind] += 1
            m = m2
            if touches_module:
                n_mod = mark_module(genome, in_mod, mod_list, n_mod, stack)
            st[7] += 1
            st[8] += 2 * n_mod + 1
            if do_fb and _sm_uniform(rng) < p_res:
                fb, tepoch, depoch, epoch2 = _resident_fb(
                    genome, target, m, mod_list, n_mod, k_edges, rng,
                    tt, tstamp, tepoch, ptr, adj, fill, dstamp, depoch,
                    memo2, stamp2, epoch2, stack, queue)
                st[9] += fb
                st[10] += 1
        else:
            st[4 + kind] += 1
            genome[e] = old
    st[0] = M
    st[11] = 1.0 if m < 4 else 0.0
    return M


@numba.njit(cache=True)
def matches_(genome, target, memo, stamp, epoch, stack):
    out = genome[len(genome) - 1]
    v = node_value(genome, out, memo, stamp, epoch, stack)
    return POPCOUNT4[(~(v ^ target)) & 0xF]


@numba.njit(cache=True)
def run_batch_obs(genome0, target, pfix, m_max, seed, n_trials, p_res, k_edges):
    """Like core.run_batch, plus stats[n_trials, len(STAT_COLS)]."""
    counts = np.zeros((n_trials, 5), dtype=np.int64)
    stats = np.zeros((n_trials, 12), dtype=np.float64)
    rng = np.zeros(1, dtype=np.uint64)
    for j in range(n_trials):
        np.random.seed(seed + j)
        rng[0] = np.uint64(seed + j) * np.uint64(0x2545F4914F6CDD1D) + np.uint64(12345)
        g = genome0.copy()
        _run_one_obs(g, target, pfix, m_max, counts[j], stats[j], p_res, k_edges, rng)
    return stats, counts


# ── Python-facing helpers ─────────────────────────────────────────────────────

def run_observed(f, Np, genome, n_trials=1, seed=0, m_max=10**9, p_res=1.0,
                 k_edges=10**9, random_walk=False):
    """
    Instrumented runs. Returns (stats, counts); stats columns are STAT_COLS.
    f is a goals.py function or a 4-bit truth table (int). random_walk=True
    accepts every mutation (Fig. S1). p_res = 0 switches f_b off.
    """
    tgt = f if isinstance(f, (int, np.integer)) else target_bits(f)
    pf = np.ones(9) if random_walk else pfix_table(Np)
    g0 = np.asarray(genome, dtype=np.int64)
    return run_batch_obs(g0, tgt, pf, m_max, seed, n_trials, float(p_res), int(k_edges))


def _module(genome):
    g = np.asarray(genome, dtype=np.int64)
    n_nodes = (len(g) - 1) // 2 + 2
    in_mod = np.zeros(n_nodes, dtype=np.bool_)
    mod_list = np.empty(n_nodes, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    n = mark_module(g, in_mod, mod_list, 0, stack)
    return g, mod_list[:n].copy()


def computing_edges(genome) -> int:
    """Number of computing edges (on an input -> output path): 2 |module| + 1."""
    _, mod = _module(genome)
    return 2 * len(mod) + 1


def p_c(genome) -> float:
    """P(c): fraction of the 2N + 1 edges that are computing."""
    return computing_edges(genome) / len(genome)


def p_b_given_c(genome, f, k_edges=10**9, seed=0) -> float:
    """
    P(b|c) of one genotype, as in Methods 6.5: mean over computing edges of
    (#beneficial alternatives) / (#allowed alternatives). Exact unless
    k_edges < number of computing edges (then a uniform edge sample).
    """
    tgt = f if isinstance(f, (int, np.integer)) else target_bits(f)
    g, mod = _module(genome)
    n_nodes = (len(g) - 1) // 2 + 2
    z = lambda k=n_nodes: np.zeros(k, dtype=np.int64)
    stack = np.empty(2 * n_nodes + 4, dtype=np.int64)
    memo, st = z(), z()
    m0 = matches_(g, tgt, memo, st, 1, stack)
    rng = np.array([seed], dtype=np.uint64)
    fb, *_ = _resident_fb(g, tgt, m0, mod, len(mod), int(k_edges), rng,
                          z(), z(), 0, z(n_nodes + 1), z(2 * n_nodes), z(), z(), 0,
                          z(), z(), 0, stack, np.empty(n_nodes, dtype=np.int64))
    return fb


def warmup():
    g = np.array([0, 1, 0, 1, 0, 1, 2], dtype=np.int64)
    run_batch_obs(g, 0b1000, pfix_table(1e4), 1000, 0, 2, 1.0, 3)
    run_batch_obs(g, 0b1000, pfix_table(1e4), 1000, 0, 2, 1.0, 10**9)
    p_b_given_c(g, 0b1000)
