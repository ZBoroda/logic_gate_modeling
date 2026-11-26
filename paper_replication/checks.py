"""
Pass/fail checks of the rerun data against exact theory and against the claims
the figure titles make (see CHANGES.md). Run before regenerating the figures:

  python -m paper_replication.checks            # prints a report, exit 1 on failure

A title that depends on a failed check must be changed, not the check. A check
is changed only when its expectation was wrong; CHANGES.md records every such case.
"""
import io
BASE = '80cc677'   # last commit before the rerun: the regression reference
import os
import subprocess
import sys

import numpy as np

from paper_replication.figures import load, pooled, series, grid, par_theory, PI_H
from paper_replication.run import EXPS, path, ANDNOT
from bottleneck_replication.theory import exact_M_parallel_terms, enumerate_p
from bottleneck_replication.starts import parallel

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}')


def slope(r, lo, hi):
    m = (r[:, 0] >= lo) & (r[:, 0] <= hi) & (r[:, 1] > 0)
    return np.polyfit(np.log(r[m, 0]), np.log(r[m, 1]), 1)[0] if m.sum() >= 3 else np.nan


def near(r, theory):
    """Theory against the plain 95% CI of every row. With n rows, ~5% miss by chance:
    pass if the misses are no more than a Binomial(n, 0.05) count exceeds with p < 0.01."""
    from scipy.stats import binom
    bad = []
    for N, v, _, _, lo, hi, _ in r:
        t = theory(N)
        if not (lo <= t <= hi):
            bad.append(f'N={int(N)}: {v:.4g} [{lo:.4g}, {hi:.4g}] vs {t:.4g}')
    k = int(binom.ppf(0.99, len(r), 0.05))
    return len(bad) <= k, (bad + [f'{len(bad)} of {len(r)} outside (≤ {k} allowed)']) if bad else []


def regression():
    """Same seeds: everything the old engine recorded must be unchanged."""
    n = bad = 0
    for exp in EXPS:
        for N in grid(exp, 10**7):
            rel = 'results/' + os.path.basename(path(exp, N))
            try:
                blob = subprocess.check_output(['git', 'show', f'{BASE}:paper_replication/{rel}'],
                                               stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue
            old = np.load(io.BytesIO(blob))
            new = np.load(path(exp, N))
            n += 1
            same = (np.array_equal(old['stats'][:, :8], new['stats'][:, :8])
                    and np.array_equal(old['counts'], new['counts']))
            bad += not same
            if not same:
                print(f'    differs: {exp} N={N}')
    check('regression: M, fate counts, residents, counts unchanged vs ' + BASE, n and not bad,
          f'{n - bad}/{n} files identical')
    viol = 0
    for exp in EXPS:
        for N in grid(exp, 10**7):
            d = load(exp, N)
            if np.isfinite(d['n_comp']).all():
                viol += int(((d['n_ben'] > d['n_comp']) | (d['n_comp'] > d['M'])).sum())
    check('invariant n_ben <= n_comp <= M', viol == 0, f'{viol} violating runs')


def parallel_1e4():
    th = lambda k: (lambda N: float(par_theory(N)[k]))
    big = lambda r: r[r[:, 0] >= 10**4]
    for lab, args, k in [('(2N+1) P(c) -> 13/3', ('n_comp', 'M'), 'pc2'),
                         ('P(b|c) -> 4/13', ('n_ben', 'n_comp'), 'pbc'),
                         ('P_fix -> pi(1/2)', ('fix_ben', 'n_ben'), 'pfix')]:
        r = pooled('par_1e4', *args, nmax=10**7,
                   scale=(lambda n: 2 * n + 1) if k == 'pc2' else None)
        if not len(big(r)):
            check(f'par_1e4 pooled {lab} (N >= 1e4)', False, 'no data with n_comp')
            continue
        ok, bad = near(big(r), th(k))
        check(f'par_1e4 pooled {lab} (N >= 1e4)', ok and len(big(r)),
              f'{len(big(r))} N; ' + ('; '.join(bad) if bad else
                                      f'e.g. {big(r)[-1, 1]:.4f} vs {th(k)(big(r)[-1, 0]):.4f}'))
    pr = []
    for N in grid('par_1e4', 10**7):
        d = load('par_1e4', N)
        if N >= 10**4 and np.isfinite(d['pbc_mut']).any():
            pr.append(np.nanmean(d['pbc_mut']))
    if pr:
        print(f'    (per-run mean P(b|c), N >= 1e4: {np.mean(pr):.3f}; rejected estimator, expect ~0.40)')
    fb = series('par_1e4', 'fix_ben', nmax=10**7)
    m = fb[fb[:, 0] >= 400]
    check('par_1e4 fixed beneficial = 2 (N >= 400)', ((m[:, 1] >= 2) & (m[:, 1] <= 2.01)).all(),
          f'{m[:, 1].min():.3f}–{m[:, 1].max():.3f}')
    nb = series('par_1e4', 'n_ben', nmax=10**7)
    m = nb[nb[:, 0] >= 1000]
    ok, bad = near(m, th('n_ben'))
    check('par_1e4 beneficial proposals match theory (N >= 1e3)', ok, '; '.join(bad) or
          f'{m[:, 1].min():.2f}–{m[:, 1].max():.2f} (2/pi(1/2) = {2 / PI_H:.2f})')
    fd = sum(load('par_1e4', N)['fix_del'].sum() for N in grid('par_1e4', 10**7))
    check('par_1e4 no deleterious fixation', fd == 0, f'{fd:.0f} in total')
    rows = []
    for N in grid('par_1e4', 10**7):
        if N >= 10**4:
            d = load('par_1e4', N)
            rows.append(d['fix_neu'].mean() / (d['M'].mean() / 1e4))
    # pi(0) = 1/N_P exactly, so this restates the neutral share of proposals: printed only
    print('    (par_1e4 neutral fixations / (<M>/N_P), N >= 1e4: '
          + ', '.join(f'{x:.3f}' for x in rows) + '; identity)')
    # structural f_b and computing edges: crossover from 3/8 and 5 to 4/13 and 13/3
    rw = pooled('par_1e4', 'fb_sum', 'fb_n', w='w_fb', nmax=10**7)
    rw = rw[rw[:, 0] >= 40]                       # theory ignores the small-N detour via fitness 1/4
    ok, bad = near(rw, th('fb'))
    check('par_1e4 structural f_b (pooled, thinning-weighted) follows 3/8 -> 4/13 (N >= 40)', ok and len(rw),
          '; '.join(bad) or ', '.join(f'{int(N)}: {v:.3f}' for N, v, *_ in rw[::3]))
    ru = pooled('par_1e4', 'fb_sum', 'fb_n', nmax=10**7)
    if len(ru):
        print(f'    (unweighted pooled f_b at N = {int(ru[-1, 0]):,}: {ru[-1, 1]:.3f}; thinning artefact ~0.328)')
    rc = pooled('par_1e4', 'ce_sum', 'n_res', nmax=10**7)
    rc = rc[rc[:, 0] >= 40]                       # theory ignores the small-N detour via fitness 1/4
    ok, bad = near(rc, th('ce'))
    check('par_1e4 computing edges per resident (pooled) follow 5 -> 13/3 (N >= 40)', ok,
          '; '.join(bad) or f'{rc[0, 1]:.3f} … {rc[-1, 1]:.3f}')
    # M: exact Eq. 3, shape, stages
    chi = []
    for N in grid('par_1e4', 10**7):
        M = load('par_1e4', N)['M']
        t1, t2 = exact_M_parallel_terms(N)
        chi.append(((M.mean() - t1 - t2) / (M.std(ddof=1) / np.sqrt(len(M)))) ** 2)
    check('par_1e4 <M> vs exact Eq. 3', np.mean(chi) < 1.6, f'chi2/dof = {np.mean(chi):.2f} on {len(chi)} N')
    cv, mm, stage = [], [], []
    for N in grid('par_1e4', 10**7):
        d = load('par_1e4', N)
        M = d['M']
        t1, t2 = exact_M_parallel_terms(N)
        se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
        stage.append(max(abs(d['M0'].mean() - t1) / se(d['M0']), abs(d['Mh'].mean() - t2) / se(d['Mh'])))
        if N >= 1000:
            cv.append(M.std() / M.mean()); mm.append(np.median(M) / M.mean())
    # limits sqrt(5)/3 and 0.819 (two exponential waits, ratio 1:2) +- 3 sampling SE at 1000 runs
    check('par_1e4 shape (N >= 1e3): CV in 0.745 ± 0.06, median/mean in 0.819 ± 0.045',
          all(abs(x - 0.745) <= 0.06 for x in cv) and all(abs(x - 0.819) <= 0.045 for x in mm),
          f'CV {min(cv):.3f}–{max(cv):.3f}, median/mean {min(mm):.3f}–{max(mm):.3f}')
    check('par_1e4 stages: below-1/2 and at-1/2 means within 3 SEM of the Eq. 3 terms',
          max(stage) < 3, f'max {max(stage):.2f} SEM')


def par_10():
    med = series('par_10', 'pc_mut', stat='median', nmax=10**7)
    m = med[(med[:, 0] >= 1000)]
    if len(m):
        x = m[:, 0] * m[:, 1]
        check('par_10 typical-run P(c) ∝ 1/N (N·median flat within ±20%, N >= 1e3)',
              x.max() / x.min() < 1.5, f'N·median {x.min():.2f}–{x.max():.2f}')
    pc = pooled('par_10', 'n_comp', 'M', nmax=10**7)
    pb = pooled('par_10', 'n_ben', 'n_comp', nmax=10**7)
    if len(pc):
        s1, s2 = slope(pc, 1000, 1e5), slope(pb, 1000, 1e5)
        check('par_10 pooled P(c) flat over 1e3–1e5 (|slope| <= 0.15)', abs(s1) <= 0.15, f'slope {s1:+.3f}')
        check('par_10 pooled P(b|c) ∝ 1/N over 1e3–1e5 (slope in [−1.15, −0.85])',
              -1.15 <= s2 <= -0.85, f'slope {s2:+.3f}')
    r = pooled('par_10', 'n_ben', 'M', nmax=10**7, scale=lambda n: n)
    r = r[r[:, 0] >= 100]
    check('par_10 N·P(c)P(b|c) flat at 0.75 ± 0.15 (N >= 100)', ((r[:, 1] > 0.6) & (r[:, 1] < 0.9)).all(),
          ', '.join(f'{int(N)}: {v:.2f}' for N, v, *_ in r))


def random_walk():
    pc = pooled('rw', 'n_comp', 'M', nmax=10**7)
    ce = pooled('rw', 'ce_sum', 'n_res', nmax=10**7, scale=lambda n: 1 / (2 * n + 1))
    if not len(pc):
        return
    ratio = [v / ce[ce[:, 0] == N][0, 1] for N, v, *_ in pc]
    # Computing gates sit upstream, so rewiring them closes a cycle more often and the
    # redraw moves the proposal elsewhere: per-proposal P(c) < the edge fraction.
    big = [x for (N, *_), x in zip(pc, ratio) if N >= 1000]
    check('rw pooled P(c) / computing-edge fraction = 0.50 ± 0.05 for N >= 1e3 (cycle redraws)',
          all(0.45 < x < 0.55 for x in big), f'ratio {min(big):.3f}–{max(big):.3f} (all N: {min(ratio):.3f}–{max(ratio):.3f})')
    s = slope(pc, 1000, 1e5)
    check('rw pooled P(c) nearly constant over 1e3–1e5 (|slope| < 0.06)', abs(s) < 0.06,
          f'slope {s:+.3f}, values {pc[pc[:, 0] >= 1000, 1].min():.3f}–{pc[pc[:, 0] >= 1000, 1].max():.3f}')
    pb = pooled('rw', 'n_ben', 'n_comp', nmax=10**7, scale=lambda n: n)
    m = pb[pb[:, 0] >= 1000]
    s = slope(pooled('rw', 'n_ben', 'n_comp', nmax=10**7), 1000, 1e5)
    check('rw pooled P(b|c) ∝ 1/N over 1e3–1e5 (slope −1 ± 0.15)', abs(s + 1) <= 0.15,
          f'slope {s:+.3f}, N·P(b|c) {m[:, 1].min():.1f}–{m[:, 1].max():.1f}')


def chain():
    med = series('chain_1e4', 'M', stat='median', nmax=10**7)
    m = med[med[:, 0] >= 21]
    check('chain median M in [20, 25] for N >= 21', ((m[:, 1] >= 20) & (m[:, 1] <= 25)).all(),
          f'{m[:, 1].min():.0f}–{m[:, 1].max():.0f}')
    Ns = [N for N in grid('chain_1e4', 10**7) if N >= 10**4]
    rng = np.random.default_rng(1)
    sl = []
    for _ in range(500):
        ys = [np.median(rng.choice(load('chain_1e4', N)['M'], len(load('chain_1e4', N)['M']))) for N in Ns]
        sl.append(np.polyfit(np.log(Ns), np.log(ys), 1)[0])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    check('chain median slope over 1e4–1e6: 95% CI contains 0', lo <= 0 <= hi, f'[{lo:+.3f}, {hi:+.3f}]')
    M = np.concatenate([load('chain_1e4', N)['M'] for N in grid('chain_1e4', 10**7) if N >= 1000])
    xs = np.array([200, 500, 1000, 2000, 5000])
    xp = np.array([x * (M > x).mean() for x in xs])
    check('chain tail x·P(M > x) within 2× for x in 200–5000', xp.max() / xp.min() < 2,
          ', '.join(f'{v:.2f}' for v in xp))
    from scipy.stats import ks_2samp
    a, b = load('chain_1e4', 1001), load('chain_1e4', 1000001)
    if a is not None and b is not None:
        p = ks_2samp(a['M'], b['M']).pvalue
        check('chain M distribution at N = 1001 vs 1e6 (KS p > 0.01)', p > 0.01, f'p = {p:.3f}')
    pm = series('chain_1e4', 'pbc_mut', stat='median', nmax=10**7)
    pp = pooled('chain_1e4', 'n_ben', 'n_comp', nmax=10**7)
    if len(pm):
        print(f'    chain P(b|c): typical run {pm[:, 1].min():.3f}–{pm[:, 1].max():.3f}; pooled '
              f'{pp[:, 1].min():.3f}–{pp[:, 1].max():.3f}; max CI width ratio '
              f'{np.max(pp[:, 5] / pp[:, 4]):.2f}')


def other_targets():
    rows = []
    for N in (40, 100, 200, 400):
        d = load('andnot_10', N)
        if d is None:
            continue
        ip = 1 / enumerate_p(parallel(N), 10.0, ANDNOT)[0]
        rows.append((N, d['M'].mean(), ip))
    check('I1 AND NOT I2: <M> < frozen-start 1/p0 (N >= 40)', all(m < ip for _, m, ip in rows),
          ', '.join(f'N={N}: {m:,.0f} vs {ip:,.0f}' for N, m, ip in rows))
    # N_P = 1e4 (Fig. S3, blue). Neutral fixations = <M>/N_P is an identity here (pi(0) = 1/N_P
    # exactly; it only restates the neutral share of proposals), so it is printed, not checked.
    from paper_replication.figures import wls_exponent
    q = []
    for N in grid('andnot_1e4', 400):
        h = load('andnot_1e4', N)['counts'][:, 2].mean()
        q.append((N, h * enumerate_p(parallel(N), 1e4, ANDNOT)[0]))
    if q:
        small = [x for N, x in q if N <= 10]
        large = [x for N, x in q if N >= 100]
        check('andnot_1e4 first step (proposals at fitness 1/2) = frozen-start 1/p0 within 5% for N <= 10, '
              'below 0.5 x 1/p0 for N >= 100 (drift)',
              all(0.95 <= x <= 1.05 for x in small) and all(x < 0.5 for x in large),
              ', '.join(f'{N}: {x:.2f}' for N, x in q))
    for tag in ('andnot', 'or', 'true'):
        exp = f'{tag}_1e4'
        Ns = grid(exp, 10**7)
        if not Ns:
            continue
        rat = [load(exp, N)['fix_neu'].mean() / (load(exp, N)['M'].mean() / 1e4) for N in Ns if N >= 1000]
        dele = sum(int(load(exp, N)['fix_del'].sum()) for N in Ns)
        noneu = sum(int((load(exp, N)['fix_neu'] < 1).sum()) for N in Ns)
        ex = wls_exponent(series(exp, 'M', nmax=10**7), 1000)
        print(f'    ({exp}: neutral fixations / (<M>/N_P), N >= 1e3: {min(rat):.3f}–{max(rat):.3f}; identity)')
        check(f'{exp}: no deleterious fixation, a neutral fixation in every run, slope in [0.85, 1.1] (N >= 1e3)',
              dele == 0 and noneu == 0 and 0.85 <= ex[0] <= 1.1,
              f'deleterious {dele}; runs without a neutral fixation {noneu}; slope {ex[0]:.3f} ± {ex[1]:.3f}')


def titles():
    """Claims in titles that the theory checks above do not cover."""
    fb = [load('par_1e4', N)['fix_ben'].mean() for N in grid('par_1e4', 10**7)]
    check('Fig 3C: beneficial fixations ≈ 2 at every N (1.95–2.15)', all(1.95 <= x <= 2.15 for x in fb),
          f'{min(fb):.3f}–{max(fb):.3f}')
    pc = pooled('chain_1e4', 'n_comp', 'M', nmax=10**7)
    pbc = pooled('chain_1e4', 'n_ben', 'n_comp', nmax=10**7)
    pf = pooled('chain_1e4', 'fix_ben', 'n_ben', nmax=10**7)
    ok = (pc[:, 1] > 0.9).all() and ((pbc[:, 1] > 0.03) & (pbc[:, 1] < 0.3)).all() \
        and ((pf[:, 1] > 0.25) & (pf[:, 1] < 0.4)).all()
    check('Fig 5F: every pooled factor O(1) (P(c) > 0.9, P(b|c) in 0.03–0.3, P_fix in 0.25–0.4)', ok,
          f'P(c) ≥ {pc[:, 1].min():.3f}, P(b|c) {pbc[:, 1].min():.3f}–{pbc[:, 1].max():.3f}, '
          f'P_fix {pf[:, 1].min():.3f}–{pf[:, 1].max():.3f}')
    med = [np.nanmedian(load('chain_10', N)['pc_mut']) for N in grid('chain_10', 10**7) if N >= 11]
    pc10 = pooled('chain_10', 'n_comp', 'M', nmax=10**7)
    check('Fig S5C: typical-run P(c) = 1 (N >= 11); pooled below 1', min(med) >= 0.999 and (pc10[:, 1] < 1).all(),
          f'median {min(med):.4f}–{max(med):.4f}; pooled {pc10[:, 1].min():.2f}–{pc10[:, 1].max():.2f}')


def alpha_family():
    """Fig. S7: starts with a chain of L ~ N^alpha gates in a parallel reservoir."""
    from bottleneck_replication import run_1e6 as BN
    from bottleneck_replication.fig_1e6 import two_step
    from paper_replication.figures import wls_exponent
    alphas = [0.2, 0.4, 0.5, 0.6, 0.75, 0.9]
    n = bad = 0
    for a in alphas:
        for N in BN.N_GRID:
            d, old = load(f'alpha_{a:.2f}', N), BN.path(a, N)
            if d is None or not os.path.exists(old):
                continue
            o = np.load(old)
            n += 1
            bad += not (np.array_equal(d['M'], o['M']) and np.array_equal(d['counts'], o['counts']))
    check('alpha family: M and counts identical to bottleneck_replication/results_1e6', n and not bad,
          f'{n - bad}/{n} files identical')
    pc_ex, m_ex = [], []
    for a in alphas:
        e = f'alpha_{a:.2f}'
        pc = wls_exponent(pooled(e, 'n_comp', 'M', nmax=10**7), 10**4)
        m = wls_exponent(series(e, 'M', nmax=10**7), 10**4)
        Ns = [N for N in grid(e, 10**7) if N >= 10**4]
        th = np.polyfit(np.log(Ns), np.log([two_step(N, a) for N in Ns]), 1)[0]
        pc_ex.append((a, pc[0])); m_ex.append((a, m[0], th))
    check('alpha family: pooled P(c) ∝ N^(alpha − 1) over 1e4–1e6 (exponent within 0.03)',
          all(abs(x - (a - 1)) <= 0.03 for a, x in pc_ex),
          ', '.join(f'{a:g}: {x:+.3f}' for a, x in pc_ex))
    # the two-step theory (first step + bottleneck escape) omits the later steps, so it
    # predicts the exponent, not the level
    check('alpha family: <M> exponent over 1e4–1e6 within 0.05 of the two-step theory',
          all(abs(x - t) <= 0.05 for _, x, t in m_ex),
          ', '.join(f'{a:g}: {x:.3f} vs {t:.3f}' for a, x, t in m_ex))


if __name__ == '__main__':
    for f in (regression, parallel_1e4, par_10, random_walk, chain, other_targets, titles, alpha_family):
        try:
            f()
        except Exception as e:
            check(f.__name__, False, f'error: {type(e).__name__}: {e}')
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f'\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed')
    sys.exit(1 if n_fail else 0)
