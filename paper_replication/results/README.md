# results

Raw simulation output behind `paper_replication/` and `paper_extended/`.
`python -m paper_replication.run` produces it with `faster_circuit.observe`.
Each file holds one experiment at one network size, and each row is one
independent run, from the start network until the target is reached.

## File names: `<experiment>_N<size>.npz`

`<size>` is N, the number of NAND gates. For example, `par_1e4_N1000.npz` is
the parallel start evolving AND at N_P = 10⁴ with N = 1000.

| experiment | start network | target | population N_P | N range | runs per file | used in |
|---|---|---|---|---|---|---|
| `par_1e4` | parallel (Fig. 3A) | AND | 10⁴ | 5 – 10⁶ | 1000 | Figs 3, 4, S2, S6A |
| `chain_1e4` | chain (Fig. 5A) | AND | 10⁴ | 5 – 10⁶ (odd N) | 1000 | Figs 5, S6B |
| `par_10` | parallel | AND | 10 | 5 – 10⁵ | 500 | Fig. S4 |
| `chain_10` | chain | AND | 10 | 5 – 10⁶ (odd N) | 500 | Fig. S5 |
| `andnot_10` | parallel | I₁ AND NOT I₂ | 10 | 5 – 10⁴ | 500 | Fig. S3A |
| `or_10` | parallel | OR | 10 | 5 – 10⁴ | 500 | Fig. S3B |
| `true_10` | parallel | TRUE | 10 | 5 – 10⁴ | 500 | Fig. S3C |
| `andnot_1e4`, `or_1e4`, `true_1e4` | parallel | as above | 10⁴ | 5 – 10⁵ | 500 | Fig. S3 (`paper_replication` only) |
| `rw` | parallel | AND | none: random walk | 5 – 10⁵ | 500 | Fig. S1 |
| `alpha_0.20` … `alpha_0.90` | chain of L ≈ N^α gates + parallel reservoir | AND | 10⁴ | 25 – 10⁶ | 1000 | Fig. S7 |

What the parts of each name mean:
- **`par` / `chain`:** the start network.
  - Parallel: N NAND gates all reading I₁ and I₂, with the output on gate 1. It
    computes NAND (fitness 0 for AND), and only 3 of its 2N + 1 edges are
    computing edges.
  - Chain: N gates in series, every edge a computing edge. The chain runs use
    odd N, because an odd chain of NANDs computes NAND.
- **`_1e4` / `_10`:** the population size N_P in the fixation probability
  π(s) = (1 − e^{−s}) / (1 − e^{−N_P s}).
- **`andnot`, `or`, `true`:** the target function; with no prefix it is AND.
- **`alpha_<α>`:** a start between the two: a chain of L ≈ N^α gates (odd,
  at least 3) feeding the output, the other N − L gates parallel
  (`bottleneck_replication.starts`). These runs use the N grid 25, 50, 100, 200,
  500, 10³ … 10⁶ and the seeds of `bottleneck_replication/run_1e6.py`, so M and
  `counts` equal `bottleneck_replication/results_1e6/a<α>_N<N>.npz` exactly.
- **`rw`:** the random walk. Every proposed mutation is accepted, whatever its
  effect on fitness.

The N grid is 5, 10, 20, 40, 100, 200, 400, 10³, 2·10³, 4·10³, 10⁴, 2·10⁴,
4·10⁴, 10⁵, 2·10⁵, 4·10⁵, 10⁶; the chain runs use the next odd number. The
N_P = 10 runs and the random walk stop earlier because, without strong
selection, the computing part of the network drifts to thousands of gates and
each mutation becomes expensive to evaluate.

## Arrays in each file

| key | shape | meaning |
|---|---|---|
| `stats` | runs × 13 | per-run counters (below) |
| `counts` | runs × 5 | `counts[:, m]` = proposals made while the resident network matched m of the 4 truth-table rows (fitness m/4). Column 4 is always 0, because the run stops at fitness 1. Row sums equal M. |
| `seconds` | scalar | total CPU time for the file |
| `r_fb` | scalar | f_b is measured on about this many residents per run (30) |
| `k_edges` | scalar | above this many computing edges (64), f_b is estimated from a sample of edges |
| `fb_def` | string | the f_b convention, `edge_pooled_no_final` (see below) |

### `stats` columns (`faster_circuit.observe.STAT_COLS`)

| col | name | meaning |
|---|---|---|
| 0 | `M` | mutations proposed until the target is reached, fixed or lost. Proposals that would close a cycle are redrawn and not counted. |
| 1 | `fix_ben` | fixed mutations that raised fitness |
| 2 | `fix_del` | fixed mutations that lowered fitness |
| 3 | `fix_neu` | fixed mutations that left fitness unchanged |
| 4 | `lost_ben` | lost (non-fixating) mutations that would have raised fitness |
| 5 | `lost_del` | lost mutations that would have lowered fitness |
| 6 | `lost_neu` | lost mutations that would have left fitness unchanged |
| 7 | `n_res` | resident networks in the run: the start plus every fixed mutant, the final one included (= 1 + fixed mutations) |
| 8 | `ce_sum` | computing edges summed over those residents. `ce_sum / n_res` is the paper's "mean number of computing edges". |
| 9 | `fb_sum` | Σ over measured residents of (computing edges × that resident's f_b) |
| 10 | `fb_n` | Σ over measured residents of their computing edges. `fb_sum / fb_n` is f_b, the fraction of a computing edge's allowed rewirings that are beneficial, pooled over edges. 0 when f_b was not measured. |
| 11 | `capped` | 1 if the run hit the limit of 10⁹ proposals without reaching the target. One run in all the files did: run 277 of `or_1e4_N100000` (M = 10⁹, stuck at fitness ¾), so the mean M of that file is a slight underestimate. |
| 12 | `n_comp` | proposals that hit a computing edge of the current resident |

Columns 1–6 sum to M. In the random walk every proposal is "fixed", so
columns 4–6 are 0.

## Derived quantities

These are computed in `figures.py`.

- **Paper's conventions** (`paper_extended/`):
  - computing edges = `ce_sum / n_res` per run, averaged over runs;
  - P(c) = `ce_sum / n_res / (2N + 1)`;
  - P(b|c) = f_b = `fb_sum / fb_n`.
- **Per-proposal Eq. 2 factors** (`paper_replication/`, pooled over runs):
  - P(c) = Σ `n_comp` / Σ M;
  - P(b|c) = Σ (`fix_ben` + `lost_ben`) / Σ `n_comp`;
  - P_fix = Σ `fix_ben` / Σ (`fix_ben` + `lost_ben`).
- **Steps of the adaptive path:** `counts[:, 0] + counts[:, 1]` is the time
  spent below fitness ½, and `counts[:, 2]` the time spent at ½ (the two terms of
  Eq. 3).

### f_b measurement

- **Which residents:** f_b is measured on the computing edges of every resident
  except the final, fitness-1 one, which is the paper's convention.
- **Thinning to save time:** each resident is measured with probability
  min(1, 30 / residents). The figures re-weight runs by max(1, `n_res` / 30)
  when pooling.
- **Large modules:** above 64 computing edges, f_b is estimated from 64 sampled
  edges.
- **Not measured:** f_b is not measured for `andnot_*`, `or_*`, `true_*` or
  `rw` (`fb_n` = 0).
- **No effect on the evolution itself:** thinning and edge sampling use a
  private random stream, so M and every fixation are bit-identical to an
  unobserved run with the same seed.

## Reproducibility

Run j of a file is seeded with `seed_base(exp, N) + j` (`run.py`). Rerunning a
file therefore reproduces it exactly, however the work is split into chunks.
Existing files are skipped; delete one to rerun it.
