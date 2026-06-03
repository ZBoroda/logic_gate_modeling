"""
Run random walk simulations for Fig S1.

Starting from the simple initial network (fig 3A — each gate takes I1 and I2
as inputs, output connected to gate 1), evolves to AND with no selection pressure.
Every mutation is accepted regardless of fitness change.

Results saved to figS1_data/. Already-computed sizes are skipped so runs can
be extended incrementally.
"""
import os
import pickle
import numpy as np

from logic_gates import Circuit, run_in_parallel_same_start
from fast_circuit import run_random_walk_fast
from fast_circuit.circuit import warmup
from goals import and_funct


def construct_genome(size: int) -> list:
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome


def load(name, out_dir):
    path = os.path.join(out_dir, name + '.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def save(data, name, out_dir):
    with open(os.path.join(out_dir, name + '.pickle'), 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    print("Warming up Numba JIT (populates disk cache for worker processes)...")
    warmup()

    out_dir = 'figS1_data'
    os.makedirs(out_dir, exist_ok=True)

    all_sizes = [3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320,
                 480, 640, 1000, 1500, 2000, 3000]
    num_trials = 500
    N_pop = 1000
    mu = 0.001
    t_max = 1e15
    m_max = 2_000_000

    # Load existing results
    existing_sizes  = load('sizes',            out_dir) or []
    all_mutations   = load('all_mutations',    out_dir) or []
    means           = load('means_mutations',  out_dir) or []
    medians         = load('medians_mutations',out_dir) or []
    stds            = load('stds_mutations',   out_dir) or []
    pct5            = load('pct5_mutations',   out_dir) or []
    pct95           = load('pct95_mutations',  out_dir) or []

    done = set(existing_sizes)
    new_sizes = [s for s in all_sizes if s not in done]
    if not new_sizes:
        print("All sizes already computed.")
    else:
        print(f"Skipping already-done sizes: {sorted(done)}")
        print(f"Running new sizes: {new_sizes}\n")

    for size in new_sizes:
        print(f"\nNetwork size N={size}")
        initial_circuit = Circuit(2, construct_genome(size))

        mutations, times, distances, mutation_types, fitness_list, computing_list = \
            run_in_parallel_same_start(
                run_random_walk_fast, num_trials, size, None,
                and_funct, N_pop, mu, initial_circuit, t_max, m_max,
            )

        arr = np.array(mutations)
        existing_sizes.append(size)
        all_mutations.append(arr.tolist())
        means.append(float(arr.mean()))
        medians.append(float(np.median(arr)))
        stds.append(float(arr.std()))
        pct5.append(float(np.percentile(arr, 5)))
        pct95.append(float(np.percentile(arr, 95)))

        print(f"  mean={means[-1]:.1f}  median={medians[-1]:.1f}  "
              f"5th={pct5[-1]:.1f}  95th={pct95[-1]:.1f}")

        # Save after each size so partial results are never lost
        order = np.argsort(existing_sizes)
        save([existing_sizes[i] for i in order],  'sizes',            out_dir)
        save([all_mutations[i]  for i in order],  'all_mutations',    out_dir)
        save([means[i]          for i in order],  'means_mutations',  out_dir)
        save([medians[i]        for i in order],  'medians_mutations',out_dir)
        save([stds[i]           for i in order],  'stds_mutations',   out_dir)
        save([pct5[i]           for i in order],  'pct5_mutations',   out_dir)
        save([pct95[i]          for i in order],  'pct95_mutations',  out_dir)
        # Re-sort in-memory lists to stay consistent
        existing_sizes = [existing_sizes[i] for i in order]
        all_mutations  = [all_mutations[i]  for i in order]
        means          = [means[i]          for i in order]
        medians        = [medians[i]        for i in order]
        stds           = [stds[i]           for i in order]
        pct5           = [pct5[i]           for i in order]
        pct95          = [pct95[i]          for i in order]

    print(f"\nDone. Results saved to {out_dir}/")
