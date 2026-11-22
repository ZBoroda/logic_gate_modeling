# paper_extended

The figures of *How many mutations does it take to evolve a new function?*
(arXiv 2610.40635) with **nothing changed except the range of network sizes**.
Every plot over network size is redrawn, with the paper's statistics, fits and
labels, from the extended `faster_circuit` runs. All other panels are the paper's
own.

For the corrected and re-analysed version of these figures, see
`paper_replication/` and its `CHANGES.md`.

```
python -m paper_extended.figures                    # -> figs_1e4/, figs_1e6/, fits_1e4.md, fits_1e6.md
python -m paper_extended.figures 1e4                # one set only
python -m paper_extended.crop_panels paper.pdf      # -> panels/ (already done)
```

| folder | contents |
|---|---|
| `figs_1e4/` | N ≤ 10⁴ (the paper's range, extended tenfold); caption numbers in `fits_1e4.md` |
| `figs_1e6/` | every simulated N (N_P = 10⁴ up to 10⁶); caption numbers in `fits_1e6.md` |

## What is extended

| figure | redrawn from new data | kept from the paper | N range in `figs_1e6` (paper: 5 – 10³) |
|---|---|---|---|
| 3 | B, C, D | A | 5 – 10⁶ |
| 4 | B, C | A, D | 5 – 10⁶ |
| 5 | B, C, E | A, D | 5 – 10⁶ |
| S1 | all | — | 5 – 10⁵ |
| S2 | C | A, B (histograms at N = 40 and 400) | 5 – 10⁶ |
| S3 | A, B, C | — | 5 – 10⁴ |
| S4 | A – D | — | 5 – 10⁵ (B: N_P = 10⁴ to 10⁶) |
| S5 | A – D | — | 5 – 10⁶ |
| S6 | A, B | — | 5 – 10⁶ |

Figs 1 and 2 have no plot over network size and are not included.

## Unchanged from the paper

- **Statistics:** each dot is the mean over runs of a per-run quantity; error bars
  are the 5th and 95th percentiles across runs (S2C: the median, no bars).
- **Definitions:**
  - computing edges are averaged over the resident networks of a run;
  - P(c) (S4C, S5C) is computing edges / (2N + 1), averaged over residents;
  - the fraction of beneficial mutations per computing edge is taken over the
    computing edges of every resident except the final one;
  - time (S6) follows Methods 6.1.
- **Fits:** red lines are y = A x^b by least squares on log(mean) against log N,
  over all points unless the caption says otherwise. The caption numbers
  (subset fits, brackets) are recomputed the same way in **`fits_1e4.md`** and
  **`fits_1e6.md`**, next to the paper's values. "Non-overlapping subsets" are contiguous halves or thirds of
  the extended N grid.
- **Layout and labels** follow the paper (MATLAB-style axes and colours).

## Data

`paper_replication/results/`: 1000 runs per N for N_P = 10⁴ (the paper used 500),
500 for N_P = 10 and the random walk. The model follows the paper's Methods; see
`paper_replication/README.md`.
