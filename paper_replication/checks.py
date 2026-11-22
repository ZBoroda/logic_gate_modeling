"""
Pass/fail checks of the rerun data against exact theory and against the claims
the figure titles make (see CHANGES.md). Run before regenerating the figures:

  python -m paper_replication.checks            # prints a report, exit 1 on failure

A title that depends on a failed check must be changed, not the check.
"""
import io
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


def near(r, theory, rel):
    """Every row: theory inside the CI widened by rel (relative)."""
    bad = []
    for N, v, _, _, lo, hi, _ in r:
        t = theory(N)
        if not (lo * (1 - rel) <= t <= hi * (1 + rel)):
            bad.append(f'N={int(N)}: {v:.4g} [{lo:.4g}, {hi:.4g}] vs {t:.4g}')
    return not bad, bad


def regression():
    """Same seeds: everything the old engine recorded must be unchanged."""
    n = bad = 0
    for exp in EXPS:
        for N in grid(exp, 10**7):
            rel = 'results/' + os.path.basename(path(exp, N))
            try:
                blob = subprocess.check_output(['git', 'show', f'HEAD:paper_replication/{rel}'],
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
    check('regression: M, fate counts, residents, counts unchanged vs HEAD', n and not bad,
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
    for lab, args, k, rel in [('(2N+1) P(c) -> 13/3', ('n_comp', 'M'), 'pc2', 0.02),
                              ('P(b|c) -> 4/13', ('n_ben', 'n_comp'), 'pbc', 0.03),
                              ('P_fix -> pi(1/2)', ('fix_ben', 'n_ben'), 'pfix', 0.03)]:
        r = pooled('par_1e4', *args, nmax=10**7,
                   scale=(lambda n: 2 * n + 1) if k == 'pc2' else None)
        if not len(big(r)):
            check(f'par_1e4 pooled {lab} (N >= 1e4)', False, 'no data with n_comp')
            continue
        ok, bad = near(big(r), th(k), rel)
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
    ok, bad = near(m, th('n_ben'), 0.06)
    check('par_1e4 beneficial proposals match theory (N >= 1e3)', ok, '; '.join(bad) or
          f'{m[:, 1].min():.2f}–{m[:, 1].max():.2f} (2/pi(1/2) = {2 / PI_H:.2f})')
    fd = sum(load('par_1e4', N)['fix_del'].sum() for N in grid('par_1e4', 10**7))
    check('par_1e4 no deleterious fixation', fd == 0, f'{fd:.0f} in total')
    rows = []
    for N in grid('par_1e4', 10**7):
        if N >= 10**4:
            d = load('par_1e4', N)
            rows.append(d['fix_neu'].mean() / (d['M'].mean() / 1e4))
    check('par_1e4 neutral fixations = <M>/N_P (N >= 1e4)', all(0.95 <= x <= 1.05 for x in rows),
          ', '.join(f'{x:.3f}' for x in rows))
    # structural f_b and computing edges: crossover from 3/8 and 5 to 4/13 and 13/3
    rw = pooled('par_1e4', 'fb_sum', 'fb_n', w='w_fb', nmax=10**7)
    rw = rw[rw[:, 0] >= 20]
    ok, bad = near(rw, th('fb'), 0.035)
    check('par_1e4 structural f_b (pooled, thinning-weighted) follows 3/8 -> 4/13', ok and len(rw),
          '; '.join(bad) or ', '.join(f'{int(N)}: {v:.3f}' for N, v, *_ in rw[::3]))
    ru = pooled('par_1e4', 'fb_sum', 'fb_n', nmax=10**7)
    if len(ru):
        print(f'    (unweighted pooled f_b at N = {int(ru[-1, 0]):,}: {ru[-1, 1]:.3f}; thinning artefact ~0.328)')
    rc = pooled('par_1e4', 'ce_sum', 'n_res', nmax=10**7)
    rc = rc[rc[:, 0] >= 20]                       # theory ignores the small-N detour via fitness 1/4
    ok, bad = near(rc, th('ce'), 0.012)
    check('par_1e4 computing edges per resident (pooled) follow 5 -> 13/3 (N >= 20)', ok,
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
    check('rw pooled P(c) below computing edges/(2N+1) (cycle redraws; ratio 0.4–0.9)',
          all(0.4 < x < 0.9 for x in ratio), f'ratio {min(ratio):.3f}–{max(ratio):.3f}')
    s = slope(pc, 1000, 1e5)
    check('rw pooled P(c) ~ constant over 1e3–4e4 (|slope| < 0.06)', abs(s) < 0.06,
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


if __name__ == '__main__':
    for f in (regression, parallel_1e4, par_10, random_walk, chain, other_targets):
        try:
            f()
        except Exception as e:
            check(f.__name__, False, f'error: {type(e).__name__}: {e}')
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f'\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed')
    sys.exit(1 if n_fail else 0)
