# paper_replication

The figures of *How many mutations does it take to evolve a new function?*
(arXiv 2610.40635), redrawn from new `faster_circuit` simulations and extended
past the paper's N ≤ 1000.

The aim is not an exact copy. Where our understanding has moved on, the figures
are corrected:
- exact theory from `bottleneck_replication/`
- per-proposal (mutation-weighted) Eq. 2 factors
- estimators suited to each distribution

**Every change from the paper, and why, is listed in [CHANGES.md](CHANGES.md).**

| folder | contents |
|---|---|
| `figs_1e4/` | every figure, N ≤ 10⁴ |
| `figs_1e6/` | every figure, all sizes simulated (N_P = 10⁴ up to N = 10⁶) |

```
python -m paper_replication.run [workers] [exp ...]   # simulations -> results/
python -m paper_replication.checks                    # pass/fail checks against theory
python -m paper_replication.figures 1e4               # -> figs_1e4/
python -m paper_replication.figures 1e6               # -> figs_1e6/
```

## What was simulated

The model follows the paper's Methods:
- SSWM Moran fixation, π(s) = (1−e^{−s})/(1−e^{−N_P s}).
- Mutations as in Methods 6.2: a uniform edge, then a uniform new source; a
  proposal that closes a cycle is redrawn, edge included.
- M counts every acyclic proposal, fixed or lost.
- No-op rewirings are excluded (commit e48ada6).

| exp | start | target | N_P | N range | runs | figures |
|---|---|---|---|---|---|---|
| `par_1e4` | parallel (Fig. 3A) | AND | 10⁴ | 5 – 10⁶ | 1000 | 3, 4, S2, S6A |
| `chain_1e4` | chain (Fig. 5A) | AND | 10⁴ | 5 – 10⁶ | 1000 | 5, S6B |
| `par_10` | parallel | AND | 10 | 5 – 10⁵ | 500 | S4 |
| `chain_10` | chain | AND | 10 | 5 – 10⁶ | 500 | S5 |
| `andnot_10`, `or_10`, `true_10` | parallel | I₁∧¬I₂, OR, TRUE | 10 | 5 – 10⁴ | 500 | S3 |
| `rw` | parallel | AND | random walk | 5 – 10⁵ | 500 | S1 |

The N_P = 10 runs and the random walk stop below 10⁶. Without strong selection,
the computing module drifts to thousands of gates in some runs. Every mutation
then costs O(module) to evaluate.

`rw_N100000.npz` was not rerun with the new `n_comp` counter, because it costs
37 CPU-hours. Its other columns are identical, since the seeds are the same.

## Observables (`faster_circuit/observe.py`, per run)

- **M, and the fates of proposals:** fixed or lost; beneficial, neutral or
  deleterious.
- **`counts[m]`:** the number of proposals made while the resident matched m of
  the 4 rows. This splits M into the steps of the adaptive path.
- **`n_comp`:** proposals that hit a computing edge of the current resident.
  - The per-proposal P(c) = Σn_comp/ΣM.
  - P(b|c) = Σn_ben/Σn_comp.
- **Computing edges per resident**, summed with the resident count, so it can be
  pooled.
- **Structural f_b:** for each computing edge, the fraction of its allowed
  rewirings that are beneficial (a property of the landscape).
  - It is pooled over the edges of every non-final resident (the paper's
    convention).
  - It is measured on about 30 residents per run, and the figures re-weight runs
    for that thinning.

## Estimators

- **Per-run counts:** the mean, with a 95% run-bootstrap CI (thick bar). On M a
  thin bar also shows the 5–95% spread across runs.
  - From the chain start a handful of runs carry most of all proposals. There
    the median is primary and the mean is drawn hollow.
- **Every ratio:** a ratio of sums over runs, with a run-bootstrap CI.
- **References:** exact theory, not power-law fits.

See CHANGES.md for the reasons. `checks.py` tests the data against every
quantitative claim in the figure titles.
