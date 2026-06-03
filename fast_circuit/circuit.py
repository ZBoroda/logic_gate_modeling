"""
Array-based circuit representation with Numba-JIT-compiled hot paths.

Genome format identical to Circuit.genome:
  [in1_g0, in2_g0, in1_g1, in2_g1, ..., output_gate_index]

Key compiled functions:
  _topo_order_numba     — O(N) Kahn's BFS using CSR adjacency
  _computing_order_numba — backward BFS to find computing subgraph
  _fitness_numba        — evaluates all 4 input pairs in one JIT call
"""
import random
import functools

import numpy as np
import numba

NUM_INPUTS = 2


# ── Numba-compiled functions ──────────────────────────────────────────────────

@numba.njit(cache=True)
def _build_csr(genome, num_inputs):
    """
    Build CSR (Compressed Sparse Row) successor adjacency in O(N).
    adj[ptr[u] : ptr[u+1]] lists every gate that takes node u as an input.
    """
    n_gates = (len(genome) - 1) // 2
    n_nodes = num_inputs + n_gates

    out_degree = np.zeros(n_nodes, dtype=np.int32)
    for g in range(n_gates):
        out_degree[genome[2 * g]] += 1
        out_degree[genome[2 * g + 1]] += 1

    ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        ptr[i + 1] = ptr[i] + out_degree[i]

    adj = np.zeros(2 * n_gates, dtype=np.int32)
    pos = np.zeros(n_nodes, dtype=np.int32)
    for i in range(n_nodes):
        pos[i] = ptr[i]

    for g in range(n_gates):
        gate = g + num_inputs
        s1 = genome[2 * g]
        s2 = genome[2 * g + 1]
        adj[pos[s1]] = gate
        pos[s1] += 1
        adj[pos[s2]] = gate
        pos[s2] += 1

    return adj, ptr


@numba.njit(cache=True)
def _topo_order_numba(genome, num_inputs):
    """
    O(N) Kahn's BFS topological sort over all nodes.
    Returns (order, order_len): order_len < n_nodes signals a cycle.
    """
    n_gates = (len(genome) - 1) // 2
    n_nodes = num_inputs + n_gates

    adj, ptr = _build_csr(genome, num_inputs)

    in_degree = np.zeros(n_nodes, dtype=np.int32)
    for g in range(n_gates):
        in_degree[g + num_inputs] += 2   # each gate has exactly 2 input edges

    queue = np.zeros(n_nodes, dtype=np.int32)
    head, tail = 0, 0
    for i in range(n_nodes):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    order = np.zeros(n_nodes, dtype=np.int32)
    order_len = 0
    while head < tail:
        u = queue[head]
        head += 1
        order[order_len] = u
        order_len += 1
        for j in range(ptr[u], ptr[u + 1]):
            v = adj[j]
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue[tail] = v
                tail += 1

    return order, order_len


@numba.njit(cache=True)
def _computing_order_numba(genome, num_inputs, full_order, full_order_len):
    """
    Backward BFS from the output gate to find computing nodes,
    then filter the full topo order to just those nodes.
    Typically 3-10 nodes regardless of N.
    """
    n_gates = (len(genome) - 1) // 2
    n_nodes = num_inputs + n_gates

    reachable = np.zeros(n_nodes, dtype=np.bool_)
    queue = np.zeros(n_nodes, dtype=np.int32)
    head, tail = 0, 0

    output = genome[-1]
    reachable[output] = True
    queue[tail] = output
    tail += 1

    while head < tail:
        node = queue[head]
        head += 1
        if node >= num_inputs:
            g = node - num_inputs
            for inp in (genome[2 * g], genome[2 * g + 1]):
                if not reachable[inp]:
                    reachable[inp] = True
                    queue[tail] = inp
                    tail += 1

    eval_order = np.zeros(n_nodes, dtype=np.int32)
    eval_len = 0
    for i in range(full_order_len):
        node = full_order[i]
        if reachable[node]:
            eval_order[eval_len] = node
            eval_len += 1

    return eval_order, eval_len


@numba.njit(cache=True)
def _evaluate_numba(genome, num_inputs, eval_order, eval_len, i1, i2):
    """Evaluate circuit output for one input pair. O(computing depth)."""
    n_nodes = num_inputs + (len(genome) - 1) // 2
    values = np.empty(n_nodes, dtype=np.bool_)
    values[0] = i1
    values[1] = i2
    for i in range(eval_len):
        node = eval_order[i]
        if node < num_inputs:
            continue
        g = node - num_inputs
        values[node] = not (values[genome[2 * g]] and values[genome[2 * g + 1]])
    return values[genome[-1]]


@numba.njit(cache=True)
def _fitness_numba(genome, num_inputs, eval_order, eval_len, target):
    """
    Fraction of the 4 input pairs where output matches target.
    target: np.bool_ array of length 4, ordered (00, 01, 10, 11).
    All 4 evaluations run inside a single JIT-compiled call.
    """
    correct = 0
    for i in range(4):
        i1 = bool(i & 2)
        i2 = bool(i & 1)
        if _evaluate_numba(genome, num_inputs, eval_order, eval_len, i1, i2) == target[i]:
            correct += 1
    return correct / 4.0


# ── one-time JIT warmup ───────────────────────────────────────────────────────

def warmup():
    """
    Force Numba to compile all JIT functions before timing-sensitive code.
    Takes ~1-2s on first run; subsequent runs use disk cache (cache=True).
    """
    g = np.array([0, 1, 2], dtype=np.int32)
    full_order, full_len = _topo_order_numba(g, NUM_INPUTS)
    eval_order, eval_len = _computing_order_numba(g, NUM_INPUTS, full_order, full_len)
    target = np.array([False, False, False, True], dtype=np.bool_)
    _fitness_numba(g, NUM_INPUTS, eval_order, eval_len, target)
    print("  Numba warmup complete.")


# ── target conversion (cached per function) ───────────────────────────────────

@functools.lru_cache(maxsize=16)
def _target_array(f):
    arr = np.zeros(4, dtype=np.bool_)
    for i in range(4):
        arr[i] = f([bool(i & 2), bool(i & 1)])
    return arr


# ── FastCircuit ───────────────────────────────────────────────────────────────

class FastCircuit:
    """
    Lightweight circuit on a plain int32 numpy array.
    No NandGate/InputGate objects; genome format identical to Circuit.genome.

    Cached per-instance:
      _full_order / _full_len  — full topo order (cycle detection)
      _eval_order / _eval_len  — computing subgraph order (evaluation)
    """

    __slots__ = ('genome', '_valid', '_full_order', '_full_len',
                 '_eval_order', '_eval_len', '_fitness_cache')

    def __init__(self, genome):
        self.genome = np.asarray(genome, dtype=np.int32)
        self._fitness_cache = {}
        self._recompute_topo()

    def _recompute_topo(self):
        n_nodes = NUM_INPUTS + (len(self.genome) - 1) // 2
        full_order, full_len = _topo_order_numba(self.genome, NUM_INPUTS)
        self._full_order = full_order
        self._full_len = full_len
        if full_len == n_nodes:
            self._valid = True
            eval_order, eval_len = _computing_order_numba(
                self.genome, NUM_INPUTS, full_order, full_len)
            self._eval_order = eval_order
            self._eval_len = eval_len
        else:
            self._valid = False
            self._eval_order = np.zeros(0, dtype=np.int32)
            self._eval_len = 0

    def has_cycle(self) -> bool:
        return not self._valid

    def fitness(self, f) -> float:
        if f not in self._fitness_cache:
            if not self._valid:
                self._fitness_cache[f] = 0.0
            else:
                self._fitness_cache[f] = float(_fitness_numba(
                    self.genome, NUM_INPUTS,
                    self._eval_order, self._eval_len,
                    _target_array(f),
                ))
        return self._fitness_cache[f]

    def duplicate(self):
        c = object.__new__(FastCircuit)
        c.genome = self.genome.copy()
        c._valid = self._valid
        c._full_order = self._full_order   # read-only until next mutation
        c._full_len = self._full_len
        c._eval_order = self._eval_order
        c._eval_len = self._eval_len
        c._fitness_cache = {}
        return c

    def mutate(self):
        """In-place point mutation with cycle rejection. Retries until acyclic."""
        n_gates = (len(self.genome) - 1) // 2
        n_all = NUM_INPUTS + n_gates
        n_nodes = n_all

        while True:
            idx = random.randrange(len(self.genome))
            if idx == len(self.genome) - 1:
                new_val = random.randrange(n_gates) + NUM_INPUTS
            else:
                new_val = random.randrange(n_all)

            old_val = int(self.genome[idx])
            self.genome[idx] = new_val

            full_order, full_len = _topo_order_numba(self.genome, NUM_INPUTS)
            if full_len == n_nodes:
                self._valid = True
                self._full_order = full_order
                self._full_len = full_len
                eval_order, eval_len = _computing_order_numba(
                    self.genome, NUM_INPUTS, full_order, full_len)
                self._eval_order = eval_order
                self._eval_len = eval_len
                self._fitness_cache = {}
                return

            self.genome[idx] = old_val   # undo cyclic mutation

    def __repr__(self):
        return f'FastCircuit({self.genome.tolist()})'
