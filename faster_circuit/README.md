# faster_circuit

A Numba-compiled implementation of the algorithm from the note *Simulating the evolution of Boolean networks quickly*. Each mutation costs O(size of the computing module) instead of O(N). From the parallel start this is flat in N.

| N | fast_circuit (µs/mutation) | faster_circuit (µs/mutation) |
|---|---|---|
| 100 | 2.6 | 0.025 |
| 1 000 | 8.4 | 0.017 |
| 10 000 | 65 | 0.027 |

(Measured with `python test_faster_circuit.py`, parallel start, N_P = 10⁴.)

## What it does differently from `fast_circuit`

See `fast_circuit/ALGORITHM.md` for the full comparison. In brief:

- **4-bit truth tables.** One integer op evaluates a gate on all four input rows (I1 = `0b1100`, I2 = `0b1010`).
- **Lazy evaluation from the output.** An iterative, memoised post-order walk visits only the output's ancestors. Memo validity uses an epoch stamp, so nothing is ever cleared in O(N).
- **Local cycle test.** Rewiring gate g to source u' closes a cycle iff g is an ancestor of u'. A DFS over u''s ancestors decides this.
- **O(1) sampling** of the new source by index arithmetic, excluding the current source and the gate itself (the Methods distribution).
- **In-place mutation**, reverted when the mutant is lost. No copies.
- **No per-step time draws.** Physical time is T | M ~ Gamma(M, τ/Λ), drawn after the run if needed.
- A whole batch of runs executes inside one `@njit` function.

**Distribution difference from `fast_circuit`:** `fast_circuit` can redraw the current source, which produces a counted no-op with probability ≈ 1/(N+1). `faster_circuit` never does, so it matches the Methods and the exact enumerations.

## Usage

```python
from faster_circuit import run_strong_selection_faster
from goals import and_funct

genome = [0, 1] * 1000 + [2]          # parallel start, N = 1000 (same format as Circuit)
M, counts = run_strong_selection_faster(and_funct, Np=1e4, genome=genome,
                                        n_trials=1000, seed=0)
M.mean()            # mean mutations to reach AND
counts[:, 0].mean() # mean mutations spent at f = 0 (first step)
```

`counts[j, k]` is the number of mutations trial j spent with k of the 4 rows correct. Trial j is seeded with `seed + j`, so results are reproducible.

## Tests

```bash
conda activate logic_gate
python test_faster_circuit.py
```

The tests check fitness against `FastCircuit` (300 random DAGs × 16 targets) and the local cycle test against `FastCircuit`'s global topological sort. They also compare simulated ⟨M⟩ and first-step means against the exact parallel-start formulas.

## Observables: computing edges and P(b|c) (`observe.py`)

`run_observed(f, Np, genome, n_trials, seed, p_res, k_edges, random_walk)` is the
same simulation as `run_batch`, bit for bit (same M, same fixations), and also
records per run, in `stats` (columns `STAT_COLS`):

- fixating / lost mutations by type (beneficial, deleterious, neutral)
- computing edges of each resident network (initial + every fixed mutant), summed
- f_b of the residents: the fraction of beneficial alternatives per computing
  edge, averaged over a resident's computing edges (Methods 6.5)

Per-genotype helpers:

```python
from faster_circuit import computing_edges, p_c, p_b_given_c
computing_edges(genome)        # 2 |module| + 1 edges on an input -> output path
p_c(genome)                    # computing edges / (2N + 1)
p_b_given_c(genome, and_funct) # exact mean f_b over computing edges
```

`p_b_given_c` does not try sources one by one. Rewiring an input of gate v only
changes v's truth table, to NAND(t, w), so the output depends only on the new
source's truth table t. It counts the allowed sources by truth table: all nodes,
minus the descendants of v (they would close a cycle), minus v and the current
source. That costs O(|desc(v)| + 16 |module|) per edge. It is tested against brute
force over 300 random DAGs (`test_observe.py`).

For large runs, `p_res = -R` measures f_b on about R uniformly chosen residents per
run, replaying the run with the same seed. `k_edges` caps the number of
computing edges per resident, which are then sampled uniformly. Both choices
keep the estimate unbiased.
