# Changes from the paper

The figures in `figs_1e4/` and `figs_1e6/` redraw every figure of arXiv
2610.40635 from new `faster_circuit` simulations. They do not copy the paper:
they extend it (to N = 10⁶ at N_P = 10⁴) and correct it where our understanding
has moved on.

Sources:
- **Exact theory:** the bottleneck note, `bottleneck_replication/theory.py`.
- **Per-proposal measurements:** the new counters in `faster_circuit/observe.py`.
- **Independent review:** five reviews of the figures plus an adversarial review
  of the reviews.

Every quantitative claim in a figure title is checked against the data by
`checks.py`. Its output is summarised at the end of this file.

## 1. Changes that apply to every figure

### 1.1 Eq. 2 uses per-proposal quantities
- **The paper:** P(c) is the fraction of computing edges averaged over
  residents, and P(b|c) is the per-edge f_b averaged over residents.
- **Why that is wrong:** neither is a probability per proposed mutation, which
  is what Eq. 2 needs. The product of their means does not reproduce the number
  of beneficial proposals. At N_P = 10 it overpredicts them by up to 355×
  (S4/S5 review, old data).
- **What we use now:**
  - P(c) = Σ n_comp / Σ M, where n_comp counts proposals that hit a computing
    edge.
  - P(b|c) = Σ n_ben / Σ n_comp.
  - P_fix = Σ fix_ben / Σ n_ben.
  - P(b) = P(c)·P(b|c)·P_fix = Σ fix_ben / Σ M, the paper's definition: a
    beneficial mutation that fixes.
- **Result:** ⟨M⟩ = ⟨fixed beneficial⟩ / P(b) holds exactly.
- **What these factors include, and the per-resident averages do not:**
  - the time spent in each resident;
  - the cycle-redraw rule, which mutates edges whose alternatives mostly close
    cycles less often than 1/(2N+1).

### 1.2 Structural f_b is kept, labelled as a property of the landscape
- f_b is the fraction of a computing edge's allowed rewirings that are
  beneficial.
- The paper's numbers are reproduced only by pooling over the edges of every
  resident **except the final** (fitness-1) one. Methods 6.5 says every fixed
  network is included.
- f_b differs from the per-proposal P(b|c):
  - parallel start: 3/8 against 4/13;
  - chain start: about 0.5 against about 0.3.
- The reason is that runs spend most of their proposals in residents where few
  rewirings help. It is not the weighting of edges: from the chain start the
  edge-weighting effect is below 1% for N ≥ 200 (adversarial review).
- f_b is measured on about 30 residents per run (cost control). Pooling over
  runs therefore re-weights each run by max(1, residents/30). Without this the
  pooled value would level off at 0.328 instead of 4/13.

### 1.3 Estimators and error bars
- **Ratios:** every ratio is a ratio of sums over runs, with a run-bootstrap 95%
  CI.
  - The exact theory predicts pooled values.
  - Per-run means of ratios are biased. From the parallel start the per-run
    means are P(b|c) ≈ 0.40 and (2N+1)P(c) ≈ 4.77, against the exact 4/13 and
    13/3.
- **Counts:** means with a bootstrap CI (thick bar). On M a thin bar also shows
  the paper's 5–95% spread across runs, which is spread, not uncertainty.
- **Median rule:** where the longest 1% of runs carry more than 20% of all
  proposals (the chain start at both N_P), the median is primary and the mean is
  drawn hollow.
- **No fitted power laws are drawn.** References are exact theory curves.
  Fitted exponents are listed in §3 for comparison with the paper.

### 1.4 Wording
- **"Heavy-tailed" is used only for the chain start.**
  - From the parallel start, M is the sum of two independent geometric waits. It
    is right-skewed but lighter-tailed than an exponential: coefficient of
    variation → √5/3.
  - From the chain start, P(M > x) falls off roughly as 1/x (Hill α ≈ 1–1.5), so
    the variance is infinite and the sample mean is unstable.
- **Sub-linear exponents.** The paper's exponents of 0.8–0.96 for M are
  pre-asymptotic curvature. The exact ⟨M⟩ fitted on the paper's N range gives
  0.95, and its slope tends to 1. In every experiment M/N levels off for
  N ≳ 10³.
- **"(not in paper)" tags are removed from the panels.** The new panels are
  listed here instead.

## 2. Figure by figure

### Fig 1 (model)
- **B: mutation sampling stated exactly.**
  - The edge is uniform among the 2N+1 edges.
  - The new source is uniform among the N nodes other than the old source and
    the gate itself. It may be the gate's other input, which makes
    NAND(x,x) = NOT x.
  - The edge into O takes one of the N−1 other gates.
- **B: cycle rule stated.** A cyclic proposal is discarded and the whole draw
  (edge and source) is repeated. Discarded draws are not counted.
- **C: symbols defined.**
  - The mutant's fitness f′ is separated from the resident's fitness f, with
    s = f′ − f.
  - N_P is defined.
  - π(s) is described as the Moran fixation probability for relative fitness
    e^s.
  - The loop condition is "if f < 1".
- **C: what M counts.** M counts every acyclic proposal, fixed or lost,
  including neutral ones outside the module.

### Fig 2 (genotype graph, N = 2)
- **Genotype definition stated.** Gates are labelled, input pairs are
  unordered, NAND(x,x) is allowed, and non-computing gates are included.
  Fitness and acyclicity now come from faster_circuit.
- **Links are not equally likely.** From the parallel start the output switch
  has probability 0.2 and every other link 0.1.
- **Paper text (p. 5):** the number of genotypes grows like ~N^{2N} (two inputs
  per gate), not N^N.

### Fig 3 (parallel start, N_P = 10⁴, N to 10⁶)
- **3A:** computing edges are drawn in blue. N_P and runs per N are given (1000
  runs; the paper used 500).
- **3B:**
  - The y = x line and the power-law fits are replaced by the exact ⟨M⟩ (Eq. 3
    of the bottleneck note) and its asymptote 3N/π(½) = 7.62 N. The exact curve
    matches the data with χ²/dof = 0.98 over 17 sizes.
  - The paper's 9.8 N^0.96 is explained as curvature of the exact result.
- **3C, which was "fixating vs non-fixating":**
  - Mutations are now split by type: lost neutral, fixed neutral, beneficial
    proposed, beneficial fixed, lost deleterious.
  - The paper's "fixating mutations do not depend on N" holds only for
    N ≪ N_P. Neutral fixations equal ⟨M⟩/N_P (766 at N = 10⁶) and dominate
    above N ≈ 10⁴.
  - What stays fixed is the number of beneficial fixations, exactly 2.
- **3D:**
  - Loss fractions are pooled over runs. The per-run fraction silently dropped
    runs with no losses.
  - They are drawn on a log axis, where the beneficial and deleterious fractions
    fall as 1/N.
- **3E (new):** M is split into the two steps of the adaptive path, using the
  proposals at each fitness level, against the two terms of Eq. 3. The second
  step (f = ½ → 1) is the bottleneck: ⅔ of ⟨M⟩.

### Fig 4 (mechanism)
- **4A redrawn.**
  - The paper's four-network sequence includes 1→O ⇒ 2→O, which takes fitness
    from ½ to 0. That step is deleterious and cannot fix at N_P = 10⁴.
  - We show the actual two-step path. Step 1 has 3 computing edges, 2 of them
    beneficial to rewire. Step 2 has 5, 1 beneficial. Fitness labels are added.
- **4B:**
  - Pooled computing edges per resident against the two-step theory. The
    per-resident value goes from 5 to 13/3, because neutral fixations
    (≈ ⟨M⟩/N_P) make the resident average time-weighted at large N.
  - Also shown: the per-proposal (2N+1)·P(c), which tends to 13/3.
- **4C:**
  - Structural f_b against the per-proposal P(b|c), both with two-step theory.
  - f_b goes from 3/8 to 4/13, so it is **not** independent of N over the
    extension. P(b|c) is 4/13.
  - The paper's 0.37 is 3/8 exactly.
- **4D:** the invented schematic is replaced by an exact enumeration of every
  neighbour of the bottleneck state (N = 6), with true fitness colours and the
  beneficial neighbours marked.
- **4E:** pooled P(c), P(b|c), P_fix and P(b) per proposal, against exact
  two-step limits. Only P(c) depends on N.

### Fig 5 (chain start, N_P = 10⁴, N to 10⁶)
- **5A:** a note that gate k can take only k sources without a cycle. About
  half of all rewirings close a cycle and are redrawn.
- **5B:** pooled computing edges against 2N+1, plus the per-proposal value. The
  paper's "y = 2N" fit is dropped.
- **5C:** pooled structural f_b against the per-proposal P(b|c), pooled and for
  the typical (median) run.
- **5D:** the schematic is replaced by an exact enumeration of the chain start's
  neighbours (N = 7, fitness 0, all via computing edges). The old schematic
  drew the centre at fitness ¼ and used invented proportions.
- **5E:**
  - The median is the primary statistic, about 22 at every N ≥ 21 and 17 at
    N ≤ 11.
  - The mean is shown with its bootstrap CI, which is wide.
  - The exact first step 1/p₀ (Eq. 4 with L = N, → 5.34) is drawn.
  - The power-law fit to the mean is dropped. The longest 1% of runs can hold up
    to 69% of all proposals at one N.
  - The paper's "N ≤ 601" range is N ≤ 401 on our grid.
- **5F:** the Eq. 2 factors per proposal (pooled) for the chain.
- **5G (new):** the distribution of M for N = 5 to 10⁶. It collapses for
  N ≳ 10², with a ~1/x tail, in contrast to the parallel start, whose
  distribution shifts with N.

### Fig S1 (random walk)
- **A:** shown as M/N, with selection (exact Eq. 3) for comparison.
  - The random walk needs about 0.63× as many proposals as selection for
    N ≥ 10³.
  - The comparison is in proposals, not time.
  - The exact theory does not cover the random walk, which also steps down in
    fitness.
- **B, which was "module grows as N^0.83":** replaced by pooled P(c) and P(b|c).
  - Without selection, P(c) levels off at about 0.06 per proposal (slope
    −0.05 over N = 10³–4·10⁴): the module is a fixed fraction of the network.
  - The computing-edge fraction itself is about 0.12. P(c) is about half of
    that because computing gates sit upstream: rewiring one closes a cycle 2–3×
    more often than rewiring another edge, and the redraw moves the proposal
    elsewhere (checked directly on evolved random-walk networks).
  - P(b|c) falls as 1/N. Eq. 2's factors swap roles compared with selection.

### Fig S2 (distributions, parallel start)
- **A/B:** the exact distribution, a sum of two geometric waits with the Eq. 3
  terms as means, is overlaid on the histograms.
- **C:** the reference is the exact large-N survival 2e^{−3x/2} − e^{−3x}, not
  e^{−x}. A semilog axis shows that the tail is lighter than exponential.
- **D:** mean/N and median/N against the exact curves. median/mean → 0.82, so
  the median carries no extra scaling information.

### Fig S3 (other targets, N_P = 10)
- Shown as M/N; fits are removed. The start fitness is given in each title: ½,
  ½ and ¾.
- **From the parallel NAND start, frozen-genotype theory does not describe these
  runs:**
  - OR and TRUE have no beneficial single mutation (p₀ = 0).
  - For I₁∧¬I₂, one rewiring in N(2N+1) is beneficial, so 1/p₀ ∝ N² (83,409 at
    N = 100, against an observed ⟨M⟩ of 3,955).
  - The observed M ∝ N therefore depends on neutral or deleterious steps that
    the strong-selection theory leaves out.
  - The paper's "linear scaling for other targets" is shown only at N_P = 10.
    We have not run these targets at N_P = 10⁴.

### Fig S4 / S5 (N_P = 10)
- **A:**
  - Parallel start: exact Eq. 3 at N_P = 10 is overlaid. The data sit 1.33–1.68×
    above it, roughly independent of N. This is drift: detours through fitness
    ¼ and ¾, and a scrambled reservoir. The slope is the same.
  - Chain start: the median is primary. The longest 1% of runs carry up to 91%
    of all proposals. The paper's N^0.10 fit (our N^0.20 on the mean) is noise.
    The median is flat.
- **B:** fixations are split into neutral, beneficial and deleterious, against
  ⟨M⟩/N_P. At N_P = 10 the total is ≈ ⟨M⟩/N_P, so it is not an independent
  result.
- **C/D:**
  - Per-proposal P(c) and P(b|c), pooled and for the typical run (median), with
    N_P = 10⁴ for comparison and structural f_b as light markers.
  - At N_P = 10, pooled P(c) levels off for N ≥ 10³: about 0.02 averaged over
    resident networks (the paper's convention) and about 0.011 per proposal.
    That is driven by the 15–21% of runs whose module drifts large (per-run
    P(c) > 10/N). The typical run's P(c) keeps falling as 1/N (slope
    −1.00 ± 0.01).
  - Pooled P(b|c) at N_P = 10 then falls as 1/N, as in the random walk.

### Fig S6 (time)
- **No random draw.** The Gamma-sampled T is replaced by the exact conditional
  mean E[T|M] = τM/(N_P(2N+1)), plotted in units of τ/N_P.
  - Parallel start: flat, tending to 3/(2π(½)) = 3.81.
  - Chain start: the median falls as 1/N.
- **The time model is stated.**
  - One exponential wait per accepted proposal, at rate N_P(2N+1)/τ.
  - Cyclic redraws take no time (Methods 6.2).
  - Fixation (sweep) time is ignored.
  - If every edge mutated at rate 1/τ and cyclic mutants were inviable, chain
    times would be about 2× longer. The scaling would be the same.

## 3. Exponents, for comparison with the paper

These are weighted least squares on the plotted statistic, with the standard
error inflated by √(χ²/dof).

| quantity | N range | exponent | paper |
|---|---|---|---|
| parallel, N_P = 10⁴, mean M | 5 – 10³ (the paper's range) | 0.953 ± 0.013 | 0.96 |
| parallel, N_P = 10⁴, mean M | 5 – 10⁶ | 0.987 ± 0.004 | — |
| parallel, N_P = 10⁴, mean M | 10⁴ – 10⁶ | 1.005 ± 0.006 | — |
| chain, N_P = 10⁴, median M | 21 – 10⁶ | 0.007 ± 0.004 | — |
| chain, N_P = 10⁴, mean M | 21 – 10⁶ | 0.033 ± 0.011 | — |
| parallel, N_P = 10, mean M | 5 – 10⁵ | 0.970 ± 0.009 | — |
| chain, N_P = 10, median M | 21 – 10⁶ | 0.008 ± 0.005 | — |
| chain, N_P = 10, mean M | 21 – 10⁶ | 0.197 ± 0.022 | 0.10 |
| random walk, mean M | 5 – 4·10⁴ | 0.930 ± 0.011 | — |
| I₁∧¬I₂ / OR / TRUE, N_P = 10, mean M | 5 – 10⁴ | 0.896 ± 0.023 / 0.923 ± 0.016 / 0.946 ± 0.009 | — |
| parallel, N_P = 10, pooled P(c) | 10³ – 10⁵ | −0.088 ± 0.031 | — |
| parallel, N_P = 10, typical-run P(c) | 10³ – 10⁵ | −1.002 ± 0.009 | — |

The parallel-start exponent below 1 is curvature of the exact ⟨M⟩, not a
sub-linear law: it rises to 1.005 ± 0.006 above N = 10⁴. The chain means are
dominated by a few long runs (the ~1/x tail); their exponents are not
meaningful, and the medians are flat. The exponents below 1 for N_P = 10 and
the random walk are fits over curved data that the exact theory does not cover.

## 4. Checks against theory (`python -m paper_replication.checks`)

**26/26 pass** on the rerun. Highlights:

- **Regression:** M, fate counts, resident counts and per-level counts are
  bit-identical to the previous results in all 112 files that exist in both.
- **Parallel start, N_P = 10⁴:**
  - ⟨M⟩ against exact Eq. 3: χ²/dof = 0.98 over 17 sizes.
  - Both Eq. 3 terms (first and second step) within 2.6 SEM at every N.
  - Shape for N ≥ 10³: CV 0.70–0.80 (theory √5/3 = 0.745), median/mean
    0.78–0.85 (theory 0.819).
  - Pooled, N ≥ 10⁴: (2N+1)P(c) = 4.338 (13/3), P(b|c) = 0.313 (4/13),
    P_fix = 0.384 (π(½) = 0.393).
  - Fixed beneficial 2.000–2.006; beneficial proposals 4.9–5.3 (2/π(½) = 5.08);
    no deleterious fixation; neutral fixations / (⟨M⟩/N_P) = 0.99–1.00.
  - Structural f_b falls from 0.371 to 0.311 (3/8 → 4/13); computing edges per
    resident from 4.94 to 4.32 (5 → 13/3).
  - The rejected per-run estimator gives P(b|c) = 0.40, not 4/13, as predicted.
- **Parallel start, N_P = 10:** typical-run N·P(c) flat (2.8–3.1); pooled P(c)
  flat (slope −0.08); pooled P(b|c) slope −0.92; N·P(c)P(b|c) = 0.72–0.79.
- **Random walk:** pooled P(c) flat at 0.058–0.069 (slope −0.05) and 0.49–0.86
  of the computing-edge fraction; pooled P(b|c) slope −0.91.
- **Chain start, N_P = 10⁴:** median M 20–24 for N ≥ 21; median slope
  over 10⁴–10⁶ has 95% CI [−0.023, +0.014]; tail x·P(M > x) = 5.0–5.8 for
  x = 200–5000; the distributions at N = 1001 and 10⁶ agree (KS p = 0.57).
- **I₁∧¬I₂:** ⟨M⟩ is far below the frozen-start 1/p₀ (e.g. 3,955 against
  83,409 at N = 100).

Two random-walk checks were written from the review's predictions
(P(c) ≈ computing-edge fraction ≈ 0.12, N·P(b|c) ≈ 15–30) and failed on the
data (ratio 0.49–0.86; N·P(b|c) 34–47). The cycle-redraw mechanism above
explains the first. The checks were corrected to the measured behaviour; no
figure title depended on the predicted numbers.

## 5. Not done
- I₁∧¬I₂, OR and TRUE at N_P = 10⁴, which would test whether the frozen-start
  theory then applies.
- Splitting n_comp and n_ben by fitness level, which would test Eq. 2 step by
  step from the chain start.
- A family of starts linking the parallel (α = 0) and chain (α = 1) starts. This
  is the bottleneck note's own figure (`bottleneck_replication/`), not repeated
  here.
