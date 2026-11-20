# How `fast_circuit` simulates evolution

This note describes the algorithm in `fast_circuit/circuit.py` and `fast_circuit/models.py`. It follows the layout of the companion note *Simulating the evolution of Boolean networks quickly* (`algorithm_note.pdf`), so the two can be compared section by section.

**Summary.** Both codes simulate the same SSWM Moran process on NAND networks. The algorithms differ. `fast_circuit` is a Numba-compiled version of the literal Methods recipe: on every proposal it rebuilds the successor adjacency, runs a global topological sort to detect cycles, and re-traces the computing module. That makes each proposal O(N). The note's algorithm uses local cycle tests and lazy evaluation and is O(1) per proposal for the parallel start. Because Numba is fast, our absolute cost is still low (about 3–10 µs per mutation for N = 10–1000), but it grows with N. There is also one small difference in the mutation distribution: we count no-op mutations. See §4.

---

## 1 What is simulated

The genotype is a DAG made of two inputs, N two-input NAND gates, and an output pointer to one gate. There are 2N + 1 edges. Fitness is the fraction of the 4 input rows on which the output matches the target.

**Mutation (as implemented).** Pick one of the 2N + 1 genome positions uniformly at random.

- **Gate input slot:** the new source is drawn uniformly from **all** N + 2 nodes (both inputs and every gate, including the gate itself and the current source).
- **Output slot:** the new gate is drawn uniformly from **all** N gates, including the current one.

If the result has a cycle, the change is undone and a new (position, source) pair is drawn. Rejected proposals are not counted. A self-loop always creates a cycle, so it is always rejected. **Redrawing the current source is not rejected.** It counts as a mutation that leaves the genotype unchanged (see §4).

**Population dynamics.** `run_strong_selection_fast` uses the Moran fixation probability with population size N_P:

    π(s) = (1 − e^{−s}) / (1 − e^{−N_P s}),     π(0) = 1/N_P

with overflow guards: π = 0 when −N_P s > 250, and π = 1 − e^{−s} when N_P s > 250. `run_random_walk_fast` (used for Fig S1) accepts every mutation. Both stop at f = 1 or at the caps `t_max` and `m_max`.

## 2 Data structures

**Genome as a flat `int32` array.** The genome has the same layout as `Circuit.genome`:

    [in1_g0, in2_g0, in1_g1, in2_g1, ..., in1_g(N-1), in2_g(N-1), out]

Nodes 0 and 1 are the inputs and gates are nodes 2 … N+1. Genome position `2g + slot` is input `slot` of gate `g`, and the last position is the output edge. Drawing an edge therefore takes one random integer. This matches the note's `e = 2g + slot` numbering.

**Derived structures, rebuilt on every proposal:**

- `adj, ptr`: CSR **successor** (children) adjacency, built in O(N) by `_build_csr`.
- `full_order`: a topological order of all N + 2 nodes from Kahn's algorithm. If it is shorter than N + 2, the network has a cycle.
- `eval_order`: the computing module (the ancestors of `out`) listed in topological order.

**Truth values as booleans.** Each input row is evaluated separately into a `bool` array of length N + 2. All 4 rows are evaluated inside one JIT call.

## 3 The algorithm

### Algorithm 1: one evolutionary run (`run_strong_selection_fast`)

```
Input:  target f*, population size N_P, mutation rate μ, initial genome x, caps t_max, m_max
Output: time series T (len(T) − 1 = number of mutations M), final genome

 1: x ← copy(x);  F ← Fitness(x);  T ← [0];  L ← len(x.genome) = 2N+1
 2: while T[-1] < t_max and len(T) < m_max:
 3:     τ ← Exponential(mean = 1 / (N_P μ L))          ▷ one exponential per mutation
 4:     x' ← Duplicate(x)                               ▷ O(N) genome copy
 5:     Mutate(x')                                      ▷ Algorithm 2; O(N) per proposal
 6:     F' ← Fitness(x')
 7:     if Uniform(0,1) < π(F' − F):  x ← x';  F ← F'   ▷ fixes; otherwise x' is discarded
 8:     T.append(T[-1] + τ)
 9:     if F = 1: break
10: return T, x
```

`run_random_walk_fast` is the same loop with line 7 replaced by `x ← x'; F ← F'` (every mutation is accepted).

### Algorithm 2: subroutines (`FastCircuit`)

```
 1: function Mutate(x)                                  ▷ in place on x.genome, retry until acyclic
 2:     loop
 3:         i ← uniform{0, …, 2N}
 4:         if i = 2N:  v ← 2 + uniform{0, …, N−1}      ▷ any gate, may equal current
 5:         else:       v ← uniform{0, …, N+1}          ▷ any node, may be self or current
 6:         old ← genome[i];  genome[i] ← v
 7:         (order, len) ← TopoOrder(genome)            ▷ global cycle test, O(N)
 8:         if len = N + 2:
 9:             eval_order ← ComputingOrder(genome, order)
10:             clear fitness cache;  return
11:         genome[i] ← old                             ▷ cyclic: undo and redraw

12: function TopoOrder(genome)                          ▷ Kahn's BFS, O(N)
13:     (adj, ptr) ← BuildCSR(genome)                   ▷ children lists, O(N)
14:     indeg[gate] ← 2 for every gate;  indeg[input] ← 0
15:     queue ← all nodes with indeg 0;  order ← []
16:     while queue not empty:
17:         u ← pop_front(queue);  append u to order
18:         for each child v of u:  indeg[v] −= 1;  if indeg[v] = 0: push v
19:     return order, |order|                           ▷ |order| < N+2  ⇔  cycle

20: function ComputingOrder(genome, order)              ▷ O(N)
21:     reachable ← backward BFS from out over parent edges
22:     return [n for n in order if reachable[n]]       ▷ filter the full order, O(N)

23: function Fitness(x, f*)                             ▷ O(n_c) per row, 4 rows
24:     correct ← 0
25:     for row (i1, i2) in (00, 01, 10, 11):
26:         val[0] ← i1;  val[1] ← i2
27:         for n in eval_order:  val[n] ← ¬(val[a_n] ∧ val[b_n])
28:         if val[out] = f*(i1, i2): correct += 1
29:     return correct / 4
```

## 4 Correctness, and how the distribution differs from the note

**Cycle test.** It is exact. Kahn's algorithm outputs every node exactly when the graph is acyclic.

**Fitness.** It is exact. `eval_order` is a topological order of the output's ancestors, so each computing gate is evaluated after its parents.

**Mutation distribution.** This differs from the note in one way:

| | note (Methods) | `fast_circuit` |
|---|---|---|
| gate-edge candidates | N (all nodes except the current source u, the gate itself, and the output) | N + 1 acyclic-or-not (all except the gate itself); **includes u** |
| output-edge candidates | N − 1 (gates other than current) | N (**includes current**) |
| no-op mutation counted? | never | yes, with prob. ≈ 1/(N+1) per mutation |

Measured from the parallel start, the no-op fraction is 0.089, 0.0098, and 0.0011 for N = 10, 100, and 1000. These match 1/(N+1). A no-op is neutral. Under selection it "fixes" with probability 1/N_P and leaves the state unchanged either way. Its only effect is to add one to M, so it inflates M by roughly a factor (1 + 1/N). That is negligible for large N but noticeable at the small end of Fig S1 (N = 3 to 10, about 10–30%). If exact agreement with the note's enumerations is needed, add `v ≠ old` to the draw on lines 4–5 (see §6).

**Time.** The time per step is exponential with rate N_P μ (2N+1), which does not depend on the state. That is the same Poisson clock as in the note.

## 5 Where the cost goes, compared with the note's six speed-ups

| note's choice | `fast_circuit` | cost here |
|---|---|---|
| (i) 4-bit truth tables, all rows in one op | **No.** Loops over 4 rows with a `bool` array, allocated by `np.empty(N+2)` on every call | O(n_c) per row; the allocation is O(N) |
| (ii) lazy evaluation of only the computing module | **Partly.** Evaluation touches only `eval_order`, but building it filters the full topological order | O(N) to build, O(n_c) to evaluate |
| (iii) local cycle test (ancestors of new source) | **No.** Global Kahn sort plus CSR rebuild on every proposal | O(N) |
| (iv) O(1) source sampling | **Yes**, but without excluding u (see §4) | O(1) |
| (v) mutate in place, revert on loss | **No.** `duplicate()` copies the genome every step, and a loss discards the copy | O(N) |
| (vi) time from one Gamma variate after the run | **No.** One `np.random.exponential` per mutation, appended to a Python list | O(1) per step, plus O(M) memory |
| compiled language (listed in the note's §7 as "not used") | **Yes.** Numba `@njit` | large constant-factor gain |

So per mutation we do O(N) work (CSR, Kahn, BFS, filter, copy), where the note does O(1) for the parallel start. Numba keeps the constant small:

| N | `fast_circuit` (µs/mutation)\* | note, pure Python (µs/mutation) | note's literal NetworkX version (µs/mutation) |
|---|---|---|---|
| 10 | 2.8 | 2.5 | 109 |
| 100 | 3.6 | 2.4 | 639 |
| 1000 | 9.6 | 2.4 | 6907 |

\*Measured on 20 000 mutations from the parallel start (`duplicate + mutate + fitness`, Apple Silicon, 1 core). The per-step exponential draw and list append in `models.py` are not included. The two machines differ, so compare the trend rather than the absolute numbers. Ours grows with N and the note's is flat.

**Cost per run** (parallel start, ⟨M⟩ = O(N)): O(N²) for `fast_circuit` and O(N) for the note.

## 6 Possible improvements, in order of payoff

1. **Local cycle test (iii).** Replace `TopoOrder` in `Mutate` with a DFS over the ancestors of the new source v that looks for gate g. Skip it when v is an input. This removes CSR and Kahn from the hot path.
2. **Lazy output-driven evaluation (ii) and bit-parallel truth tables (i).** Evaluate `Val(out)` recursively or with an explicit stack, memoizing over a 4-bit `uint8` per gate (I1 = 0b1100, I2 = 0b1010, `NAND = ~(a & b) & 0xF`). Fitness is then `popcount(~(val ^ target) & 0xF) / 4`. This drops `ComputingOrder` and the full topological order entirely.
3. **Mutate in place and revert (v).** Keep a single genome, write v, and restore `old` if the mutant is lost.
4. **Exclude the current source** from the draw (`v ≠ old`). This matches the Methods distribution (§4).
5. **One Gamma variate per run (vi).** Return M and draw T ~ Gamma(M, 1/(N_P μ (2N+1))) afterwards, instead of storing the full T list.

Together, (1)–(3) would make the inner loop O(n_c) and flat in N, while keeping the Numba speed. One `@njit` function could then run a whole trial.
