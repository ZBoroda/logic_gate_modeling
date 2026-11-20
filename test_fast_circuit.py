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

def exact_mutation_distribution(genome):
    """
    Exact probability of each (index, new_value) under the paper's Methods:
    uniform edge, then uniform source among the edge's L_e options (current
    source and, for gate edges, the gate itself excluded); cyclic pairs are
    redrawn, so P(pair) is proportional to 1/L_e over acyclic pairs.
    """
    n_gates = (len(genome) - 1) // 2
    probs = {}
    for idx in range(len(genome)):
        if idx == len(genome) - 1:
            opts = [v for v in range(2, n_gates + 2) if v != genome[idx]]
        else:
            opts = [v for v in range(n_gates + 2) if v not in (genome[idx], idx // 2 + 2)]
        for v in opts:
            g2 = list(genome); g2[idx] = v
            if not FastCircuit(g2).has_cycle():
                probs[(idx, v)] = 1.0 / len(opts)
    Z = sum(probs.values())
    return {k: p / Z for k, p in probs.items()}


def test_mutation_distribution(n_samples=60000, alpha=0.001):
    """Single-mutation (index, new_value) frequencies match the Methods weights exactly."""
    from scipy.stats import chisquare
    print("\n[mutation distribution] FastCircuit and Circuit vs exact Methods weights")
    genomes = [construct_genome(4),                       # parallel start
               [0, 1, 2, 2, 3, 3, 4, 4, 5],              # chain of 4 (many cyclic pairs)
               [0, 1, 0, 2, 3, 1, 2, 4, 5]]              # mixed DAG
    for genome in genomes:
        exact = exact_mutation_distribution(genome)
        keys = sorted(exact)
        for name, mutate in [
            ("FastCircuit", lambda: _fast_mutation(genome)),
            ("Circuit", lambda: _circuit_mutation(genome)),
        ]:
            random.seed(7)
            counts = dict.fromkeys(keys, 0)
            n = n_samples if name == "FastCircuit" else n_samples // 6
            for _ in range(n):
                k = mutate()
                assert k in counts, f"{name} produced invalid mutation {k} from {genome}"
                counts[k] += 1
            obs = np.array([counts[k] for k in keys])
            exp = np.array([exact[k] * n for k in keys])
            p = chisquare(obs, exp).pvalue
            print(f"  {name:11s} genome={genome}: {len(keys)} pairs, chi2 p={p:.3f}")
            assert p > alpha


def _diff(old, new):
    d = [i for i in range(len(old)) if old[i] != new[i]]
    assert len(d) == 1, (old, new)
    return d[0], int(new[d[0]])


def _fast_mutation(genome):
    x = FastCircuit(genome)
    x.mutate()
    return _diff(genome, x.genome.tolist())


def _circuit_mutation(genome):
    c = Circuit(2, list(genome))
    c.point_mutation()
    return _diff(genome, c.genome)


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
    test_mutation_distribution()
    test_rw_distribution()
    test_ss_distribution()
    test_convergence()
    benchmark()
    print("\nAll tests passed.")
