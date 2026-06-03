"""
Regression + speed tests for fast_circuit vs original implementations.

Tests:
  1. Evaluation correctness  — same genome + inputs → same output
  2. Fitness correctness      — same genome + target → same fitness
  3. RW distribution          — statistically equivalent mutation counts
  4. SS distribution          — statistically equivalent mutation counts
  5. Convergence              — all trials reach fitness 1 for both models
  6. Speed benchmark          — wall-clock comparison (informational)
"""
import random
import time
import numpy as np
from scipy.stats import ks_2samp

from logic_gates import Circuit
from logic_gates.noselectionmodel import run_random_walk
from logic_gates.strongselectionmodel import run_evolution_strong_selection
from fast_circuit import FastCircuit, run_random_walk_fast, run_strong_selection_fast
from fast_circuit.circuit import warmup
from goals import and_funct


# ── shared helpers ────────────────────────────────────────────────────────────

def construct_genome(size: int) -> list:
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome


def make_circuit(size):
    return Circuit(2, construct_genome(size))

def make_fast_circuit(size):
    return FastCircuit(construct_genome(size))


def run_trials(fn, make_init, size, n_trials, seed, **kwargs):
    """Run fn for n_trials, return list of mutation counts."""
    counts = []
    for i in range(n_trials):
        np.random.seed(seed + i)
        random.seed(seed + i)
        init = make_init(size)
        T, _, F, _, _, _ = fn(and_funct, 1000, 0.001, init, 1e15, 500_000, **kwargs)
        assert F[-1] == 1.0, f"Trial {i} did not reach fitness 1 (got {F[-1]})"
        counts.append(len(T) - 1)
    return counts


# ── test 1: evaluation correctness ───────────────────────────────────────────

def test_evaluation_correctness(n_sizes=20, n_genomes_per_size=10):
    print("\n[1] Evaluation correctness")
    from fast_circuit.circuit import _evaluate_numba

    for size in range(2, n_sizes + 2):
        for _ in range(n_genomes_per_size):
            genome = construct_genome(size)
            circuit = Circuit(2, genome)
            fc = FastCircuit(genome)

            for i in range(4):
                i1, i2 = bool(i & 2), bool(i & 1)
                orig = circuit.evaluate([i1, i2])
                fast = _evaluate_numba(fc.genome, 2,
                                       fc._eval_order, fc._eval_len, i1, i2)
                assert orig == fast, (
                    f"N={size} inputs=({i1},{i2}): orig={orig} fast={fast}\ngenome={genome}"
                )
    print(f"  PASSED — {n_sizes * n_genomes_per_size * 4} input/genome combinations matched")


# ── test 2: fitness correctness ───────────────────────────────────────────────

def test_fitness_correctness(n_sizes=20, n_genomes_per_size=10):
    print("\n[2] Fitness correctness")
    from goals import and_funct, or_funct, xor_funct

    for size in range(2, n_sizes + 2):
        for _ in range(n_genomes_per_size):
            genome = construct_genome(size)
            fc = FastCircuit(genome)

            for f in (and_funct, or_funct, xor_funct):
                # fresh Circuit per call — original has a caching bug where calling
                # evaluate_expression with a second function raises KeyError
                circuit = Circuit(2, genome)
                orig = circuit.evaluate_expression(f)
                fast = fc.fitness(f)
                assert abs(orig - fast) < 1e-9, (
                    f"N={size} {f.__name__}: orig={orig} fast={fast}\ngenome={genome}"
                )
    print(f"  PASSED — fitness values match across {n_sizes * n_genomes_per_size * 3} cases")


# ── test 3: random walk distribution ─────────────────────────────────────────

def test_rw_distribution(size=20, n_trials=200, alpha=0.01):
    print(f"\n[3] Random walk distribution (N={size}, {n_trials} trials each, α={alpha})")
    counts_orig = run_trials(run_random_walk,      make_circuit,      size, n_trials, seed=0)
    counts_fast = run_trials(run_random_walk_fast, make_fast_circuit, size, n_trials, seed=10_000)

    ks_stat, ks_p = ks_2samp(counts_orig, counts_fast)
    print(f"  orig  mean={np.mean(counts_orig):.1f}  median={np.median(counts_orig):.1f}")
    print(f"  fast  mean={np.mean(counts_fast):.1f}  median={np.median(counts_fast):.1f}")
    print(f"  KS test: stat={ks_stat:.4f}  p={ks_p:.4f}")
    assert ks_p > alpha, f"KS test rejected equivalence: p={ks_p:.4f} < {alpha}"
    print("  PASSED")


# ── test 4: strong selection distribution ────────────────────────────────────

def test_ss_distribution(size=20, n_trials=200, alpha=0.01):
    print(f"\n[4] Strong selection distribution (N={size}, {n_trials} trials each, α={alpha})")
    counts_orig = run_trials(run_evolution_strong_selection, make_circuit,      size, n_trials, seed=0)
    counts_fast = run_trials(run_strong_selection_fast,      make_fast_circuit, size, n_trials, seed=10_000)

    ks_stat, ks_p = ks_2samp(counts_orig, counts_fast)
    print(f"  orig  mean={np.mean(counts_orig):.1f}  median={np.median(counts_orig):.1f}")
    print(f"  fast  mean={np.mean(counts_fast):.1f}  median={np.median(counts_fast):.1f}")
    print(f"  KS test: stat={ks_stat:.4f}  p={ks_p:.4f}")
    assert ks_p > alpha, f"KS test rejected equivalence: p={ks_p:.4f} < {alpha}"
    print("  PASSED")


# ── test 5: convergence ───────────────────────────────────────────────────────

def test_convergence(sizes=(5, 20, 80), n_trials=30):
    print(f"\n[5] Convergence check (sizes={sizes}, {n_trials} trials each)")
    for size in sizes:
        for fn, make_init, label in [
            (run_random_walk,             make_circuit,      "rw-orig"),
            (run_random_walk_fast,        make_fast_circuit, "rw-fast"),
            (run_evolution_strong_selection, make_circuit,   "ss-orig"),
            (run_strong_selection_fast,   make_fast_circuit, "ss-fast"),
        ]:
            for i in range(n_trials):
                np.random.seed(i); random.seed(i)
                init = make_init(size)
                _, _, F, _, _, _ = fn(and_funct, 1000, 0.001, init, 1e15, 500_000)
                assert F[-1] == 1.0, f"{label} N={size} trial {i}: final fitness={F[-1]}"
        print(f"  N={size}: PASSED")
    print("  All convergence checks passed")


# ── test 6: speed benchmark ───────────────────────────────────────────────────

def benchmark(n_trials=30):
    print(f"\n[6] Speed benchmark ({n_trials} trials per config)")

    pairs = [
        ("rw  orig", run_random_walk,               make_circuit),
        ("rw  fast", run_random_walk_fast,           make_fast_circuit),
        ("ss  orig", run_evolution_strong_selection, make_circuit),
        ("ss  fast", run_strong_selection_fast,      make_fast_circuit),
    ]

    for size in (100, 500):
        print(f"\n  N={size}")
        for label, fn, make_init in pairs:
            np.random.seed(0); random.seed(0)
            t0 = time.perf_counter()
            for i in range(n_trials):
                init = make_init(size)
                fn(and_funct, 1000, 0.001, init, 1e15, 500_000)
            elapsed = time.perf_counter() - t0
            print(f"    {label}: {elapsed:.2f}s  ({elapsed/n_trials*1000:.1f}ms/trial)")


if __name__ == "__main__":
    print("Warming up Numba JIT...")
    warmup()
    test_evaluation_correctness()
    test_fitness_correctness()
    test_rw_distribution()
    test_ss_distribution()
    test_convergence()
    benchmark()
    print("\nAll tests passed.")
