"""
Regression test: verify the optimized run_random_walk produces statistically
equivalent results to the original implementation.

The original implementation is preserved inline here as `run_random_walk_orig`
so it can always be compared against, even after the source file changes.
"""
import random
import numpy as np
from collections import Counter
from scipy.spatial.distance import hamming
from scipy.stats import ks_2samp, mannwhitneyu

from logic_gates import Circuit, run_in_parallel_same_start
from logic_gates.noselectionmodel import run_random_walk
from goals import and_funct


# ── original implementation, preserved verbatim ─────────────────────────────

def run_random_walk_orig(f, N: int, mu: float, x_init: Circuit, t_max: int, m_max: int):
    p = 0
    n = 0
    T = [0.0]
    C = []
    F = [x_init.evaluate_expression(f)]
    X = [x_init]
    x = x_init
    L = len(x_init.genome)
    mutation_counter = Counter()
    while T[-1] < t_max and len(T) < m_max:
        tau_next = np.random.exponential(1 / (N * mu * L))
        xm = x.duplicate()
        xm.mutate()
        old_fitness = x.evaluate_expression(f)
        new_fitness = xm.evaluate_expression(f)
        s = (new_fitness - old_fitness)
        if s == 0:
            mutation_counter["Neutral"] += 1
        elif s > 0:
            mutation_counter["Positive"] += 1
        elif s < 0:
            mutation_counter["Negative"] += 1
        x = xm.duplicate()
        C.append(x.num_gates_computing())
        F.append(x.evaluate_expression(f))
        T.append(T[-1] + tau_next)
        if F[-1] == 1.0:
            break
    return np.array(T), x, np.array(F), mutation_counter, np.array(C), hamming(x.genome, x_init.genome) * len(x.genome)


# ── helpers ──────────────────────────────────────────────────────────────────

def construct_genome(size: int) -> list:
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome


def run_trials(fn, size, n_trials, seed):
    """Run n_trials of fn for a given network size, return list of mutation counts."""
    counts = []
    for i in range(n_trials):
        np.random.seed(seed + i)
        random.seed(seed + i)
        circuit = Circuit(2, construct_genome(size))
        T, _, F, _, _, _ = fn(and_funct, 1000, 0.001, circuit, 1e15, 500_000)
        assert F[-1] == 1.0, f"Trial {i} did not reach fitness 1 (got {F[-1]})"
        counts.append(len(T) - 1)   # number of mutation steps
    return counts


# ── tests ────────────────────────────────────────────────────────────────────

def test_identical_with_fixed_seed(size=10, n_trials=20):
    """With same seed, optimized and original should give identical mutation counts."""
    print(f"\n[1] Fixed-seed identity check (N={size}, {n_trials} trials)")
    for i in range(n_trials):
        seed = 42 + i

        np.random.seed(seed); random.seed(seed)
        circuit_a = Circuit(2, construct_genome(size))
        T_a, _, F_a, mc_a, _, _ = run_random_walk_orig(and_funct, 1000, 0.001, circuit_a, 1e15, 500_000)

        np.random.seed(seed); random.seed(seed)
        circuit_b = Circuit(2, construct_genome(size))
        T_b, _, F_b, mc_b, _, _ = run_random_walk(and_funct, 1000, 0.001, circuit_b, 1e15, 500_000)

        n_orig = len(T_a) - 1
        n_new  = len(T_b) - 1
        assert n_orig == n_new, (
            f"Trial {i}: mutation count differs — orig={n_orig}, new={n_new}"
        )
        assert mc_a == mc_b, f"Trial {i}: mutation type counters differ"

    print(f"  PASSED — all {n_trials} trials produced identical mutation counts")


def test_distribution_equivalent(size=20, n_trials=200, alpha=0.01):
    """Without fixed seed, the two distributions should be statistically indistinguishable."""
    print(f"\n[2] Distribution equivalence (N={size}, {n_trials} trials each, α={alpha})")
    counts_orig = run_trials(run_random_walk_orig, size, n_trials, seed=0)
    counts_new  = run_trials(run_random_walk,      size, n_trials, seed=10_000)

    ks_stat, ks_p = ks_2samp(counts_orig, counts_new)
    mw_stat, mw_p = mannwhitneyu(counts_orig, counts_new, alternative='two-sided')

    print(f"  orig  mean={np.mean(counts_orig):.1f}  median={np.median(counts_orig):.1f}")
    print(f"  new   mean={np.mean(counts_new):.1f}  median={np.median(counts_new):.1f}")
    print(f"  KS test:  stat={ks_stat:.4f}  p={ks_p:.4f}")
    print(f"  MW test:  stat={mw_stat:.1f}  p={mw_p:.4f}")

    assert ks_p > alpha, f"KS test rejected equivalence: p={ks_p:.4f} < {alpha}"
    assert mw_p > alpha, f"MW test rejected equivalence: p={mw_p:.4f} < {alpha}"
    print("  PASSED — distributions are statistically indistinguishable")


def test_all_reach_fitness_one(sizes=(5, 20, 80), n_trials=50):
    """All trials should converge to fitness 1 for both implementations."""
    print(f"\n[3] Convergence check across sizes {sizes}")
    for size in sizes:
        for fn, label in [(run_random_walk_orig, "orig"), (run_random_walk, "new")]:
            for i in range(n_trials):
                np.random.seed(i); random.seed(i)
                circuit = Circuit(2, construct_genome(size))
                _, _, F, _, _, _ = fn(and_funct, 1000, 0.001, circuit, 1e15, 500_000)
                assert F[-1] == 1.0, f"N={size} {label} trial {i}: final fitness={F[-1]}"
        print(f"  N={size}: PASSED")
    print("  All convergence checks passed")


if __name__ == "__main__":
    test_identical_with_fixed_seed()
    test_distribution_equivalent()
    test_all_reach_fitness_one()
    print("\nAll tests passed.")
