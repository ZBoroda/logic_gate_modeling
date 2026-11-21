# paper_replication

Every figure of *How many mutations does it take to evolve a new function?*
(arXiv 2610.40635), redrawn from new `faster_circuit` simulations and extended
past the paper's N ≤ 1000.

| folder | contents |
|---|---|
| `figs_1e4/` | every figure, N ≤ 10⁴, fits over the plotted points |
| `figs_1e6/` | every figure, all sizes simulated (N_P = 10⁴ to N = 10⁶), plus a fit over N ≥ 10⁴ |

```
python -m paper_replication.run [workers] [exp ...]   # simulations -> results/
python -m paper_replication.figures 1e4               # -> figs_1e4/
python -m paper_replication.figures 1e6               # -> figs_1e6/
```

## What was simulated

The model follows the paper's Methods:
- SSWM Moran fixation π(s) = (1−e^{−s})/(1−e^{−N_P s}).
- Mutations as in Methods 6.2.
- M counts fixed and lost mutations.

| exp | start | target | N_P | N range | runs | figures |
|---|---|---|---|---|---|---|
| `par_1e4` | parallel (Fig. 3A) | AND | 10⁴ | 5 – 10⁶ | 1000 | 3, 4, S2, S6A |
| `chain_1e4` | chain (Fig. 5A) | AND | 10⁴ | 5 – 10⁶ | 1000 | 5, S6B |
| `par_10` | parallel | AND | 10 | 5 – 10⁵ | 500 | S4 |
| `chain_10` | chain | AND | 10 | 5 – 10⁶ | 500 | S5 |
| `andnot_10`, `or_10`, `true_10` | parallel | I₁∧¬I₂, OR, TRUE | 10 | 5 – 10⁴ | 500 | S3 |
| `rw` | parallel | AND | random walk | 5 – 10⁵ | 500 | S1 |

The N_P = 10 runs and the random walk stop below 10⁶. Without strong selection
the computing module grows by drift to hundreds or thousands of gates. Every
mutation then costs O(module) to evaluate, so the larger sizes would take days.

## Observables (`faster_circuit/observe.py`)

- **Computing edges**: edges on an input → output path, which is 2|module| + 1.
  The per-run value is averaged over the resident networks: the initial one and
  every fixed mutant (Methods 6.6).
- **P(c)** = computing edges / (2N+1).
- **f_b / P(b|c)**: for each computing edge u→v, the beneficial alternatives
  divided by the allowed (acyclic) alternatives. It is averaged over the
  resident's computing edges, then over residents (Methods 6.5). The calculation
  is exact and counts candidate sources by truth table; it is tested against
  brute force in `test_observe.py`.
  - For large runs, f_b is measured on about 30 uniformly chosen residents per run.
  - If a resident has more than 64 computing edges, 64 are sampled uniformly.
  - Both shortcuts leave the estimate unbiased. They are needed for the chain
    start, which has 2N computing edges.
- **Time** (Fig. S6) is drawn after the run as T | M ~ Gamma(M, τ/(N_P(2N+1))),
  with τ = 1000.

As in the paper, the plots show means over runs, with error bars at the 5th and
95th percentiles of the per-run values.

## Comparing with the paper

- Each scaling panel draws a solid fit over the paper's own N range (N ≤ 10³,
  or 600 / 501 / 601 where the paper stopped there) and a dashed fit over the
  extension (N ≥ 10³ in `figs_1e4`, N ≥ 10⁴ in `figs_1e6`).
- The notes print the paper's numbers next to ours over the same range, using
  the paper's sensitivity checks: 3-subset mean ± SD in Figs 3B and S1, first 3 /
  last 4 points in S3, and the paper's two subsets in S4 and S5.
- Schematics follow the paper:
  - Fig 1: the paper's network, its I₁→1 ⇒ 2→1 mutation, and the branching
    flowchart.
  - Fig 2: 54 genotypes, from `gen_fig2.py`.
  - Fig 4A: the four-step mutation sequence.
  - Figs 4D and 5D: genotype-neighbourhood schematics.
- Panels that are not in the paper are marked "(not in paper)".
- **Known issue (not yet fixed):** f_b is averaged over every resident,
  including the final network that already computes AND, whose f_b is 0.
  - This pulls P(b|c) low when runs have few residents.
  - The apparent rise of P(b|c) with N comes from neutral fixations diluting
    that zero.
  - Excluding the final resident reproduces the paper's chain-start values. It
    overshoots the parallel-start values (≈0.42 vs 0.37).

## Main results (figs_1e6)

- **Fig 3**, parallel start, N_P = 10⁴: M ≈ 8.7 N^0.99, and the fit over
  N ≥ 10⁴ gives N^1.01. The near-linear scaling holds all the way to 10⁶.
- **Fig 4**: P(c) ~ N^−1.01, so Eq. 2 explains M ∝ N. The current P(b|c)
  values are affected by the known issue above.
- **Fig 5**, chain start: computing edges ≈ 2N and P(c) ≈ 1. M is about flat
  (≈ 38 N^0.04, median about 22 at every N). f_b ≈ 0.21 as currently measured
  (≈ 0.33 without the final resident, the paper's value).
- **Fig S1**, random walk (to 10⁵): M ~ N^0.93 overall and N^0.99 for N ≥ 10⁴.
  The computing module grows as N^0.83.
- **Fig S3**, other targets, N_P = 10 (to 10⁴): the exponents are about 1.0.
- **Fig S4**, parallel start, N_P = 10 (to 10⁵): M ~ N^0.97, and N^1.02 for
  N ≥ 10⁴. P(c) follows the paper's slope up to about 10³. Above about 10⁴ it
  levels off near 6×10⁻³, because drift grows the module. Eq. 2 alone no longer
  explains the scaling there.
- **Fig S5**, chain start, N_P = 10 (to 10⁶): M ~ N^0.23, with a broad spread
  across runs. P(c) ≈ 1; P(b|c) ≈ 0.17 as currently measured (see known issue).
