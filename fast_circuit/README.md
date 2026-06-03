# fast_circuit

A drop-in replacement for the `Circuit` class with Numba-JIT-compiled hot paths. Gives **50–100× speedup** over the object-based implementation, with the speedup growing with network size N.

## Why it's faster

The original `Circuit` allocates N+2 Python gate objects on every `construct_circuit()` call (which happens inside `mutate()`). At N=500 that's ~500 object creations per mutation step, and the simulation runs millions of steps.

`FastCircuit` stores the genome as a single `int32` numpy array and does all topology and evaluation work in compiled Numba functions — no Python objects, no per-step allocation.

| N | Original | FastCircuit | Speedup |
|---|----------|-------------|---------|
| 100 | ~84 ms/trial | ~1.8 ms/trial | ~47× |
| 500 | ~1800 ms/trial | ~18 ms/trial | ~100× |

## Genome format

Identical to `Circuit.genome`:

```
[in1_g0, in2_g0, in1_g1, in2_g1, ..., in1_g(N-1), in2_g(N-1), output_gate_index]
```

Node indices: `0` and `1` are the two inputs; gates are numbered `2` through `N+1`.

## Usage

```python
from fast_circuit import FastCircuit, run_random_walk_fast, run_strong_selection_fast

# construct from a genome list (same format as Circuit)
fc = FastCircuit([0, 1, 0, 1, 2])

# evaluate fitness against a target function
from goals import and_funct
print(fc.fitness(and_funct))   # float in [0.0, 1.0]

# mutate in-place (cycle-safe — retries until acyclic)
fc.mutate()

# copy without shared state
fc2 = fc.duplicate()

# run simulations — accept Circuit or FastCircuit as x_init
result = run_random_walk_fast(and_funct, N=1000, mu=0.001,
                              x_init=fc, t_max=1e15, m_max=500_000)

result = run_strong_selection_fast(and_funct, N=1000, mu=0.001,
                                   x_init=fc, t_max=1e15, m_max=500_000)
```

Both simulation functions return the same 6-tuple as the originals:
`(T, circuit, F, mutation_counter, C, final_distance)`

## First run — Numba warmup

On the first run, Numba compiles the JIT functions and caches them to disk (`__pycache__`). This takes 1–2 seconds. Subsequent runs load from cache instantly.

Call `warmup()` before spawning parallel workers so the cache is populated before subprocesses start:

```python
from fast_circuit.circuit import warmup
warmup()  # ~1-2s first time, instant after
```

## Key internals

- **`_build_csr`** — builds a Compressed Sparse Row successor adjacency in O(N)
- **`_topo_order_numba`** — Kahn's BFS topological sort; `order_len < n_nodes` signals a cycle
- **`_computing_order_numba`** — backward BFS from output gate to find the computing subgraph (typically 3–10 nodes regardless of N); evaluation only visits these nodes
- **`_fitness_numba`** — evaluates all 4 input pairs in a single JIT call

## Testing

```bash
conda activate logic_gate
python test_fast_circuit.py
```

Tests verify evaluation correctness, fitness correctness, statistical equivalence of mutation distributions (KS test), convergence, and include a speed benchmark.
