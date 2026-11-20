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
