"""
The paper's figures (arXiv 2610.40635) with nothing changed except that every
plot over network size is redrawn from the extended `faster_circuit` runs in
`paper_replication/results/`.

  python -m paper_extended.figures [1e4|1e6]   ->  paper_extended/figs_1e4/, figs_1e6/
                                                  and fits_1e4.md, fits_1e6.md (caption numbers)
  (no argument: both sets; 1e4 = N <= 10^4, 1e6 = every simulated N)

Panels that are not plots over N (the network schematics, the genotype
neighbourhoods and the fixed-N histograms of Fig. S2) are the paper's own panels,
cut from the PDF by `crop_panels.py`.

Everything is computed as in the paper:
- dots: the mean over runs of a per-run statistic; error bars: the 5th and 95th
  percentiles of that statistic across runs;
- computing edges: per run, averaged over the resident networks;
- P(c) (Figs S4C, S5C): per run, computing edges / (2N + 1), averaged over residents;
- fraction of beneficial mutations per computing edge (P(b|c), f_b): per run, over
  the computing edges of every resident except the final one;
- fractions of non-fixating mutations: per run, then averaged;
- time: Methods 6.1, M waiting times with rate N_P (2N + 1) / tau, tau = 1000;
- red lines: y = A x^b by least squares on log(mean) against log N, over the
  points the caption names (all points unless stated).
"""
import math
import os
import zlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import paper_replication.figures as pr
from paper_replication.run import EXPS

HERE = os.path.dirname(__file__)
OUT = None                                   # set by run()
TAG = None
TAU = 1000.0

# MATLAB default colours, as in the paper
BLUE, ORANGE, GREEN = '#0072BD', '#D95319', '#77AC30'
RED, BLACK = '#FF0000', '#000000'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.linewidth': 0.6, 'axes.edgecolor': '#262626',
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4, 'legend.fancybox': False, 'legend.edgecolor': '#262626',
    'legend.framealpha': 1, 'savefig.dpi': 200})

FITS = []                                    # (figure, text) lines for fits.md


# ── data ──────────────────────────────────────────────────────────────────────

def per_run(exp, N, key):
    d = pr.load(exp, N)
    with np.errstate(invalid='ignore', divide='ignore'):
        if key == 'M':
            return d['M']
        if key == 'fix':
            return d['fix']
        if key == 'lost':
            return d['lost']
        if key == 'ce':
            return d['ce_sum'] / d['n_res']
        if key == 'pc':
            return d['ce_sum'] / d['n_res'] / (2 * N + 1)
        if key == 'fb':
            return d['fb_sum'] / d['fb_n']
        if key in ('frac_ben', 'frac_del', 'frac_neu'):
            return d['lost_' + key[5:]] / d['lost']
        if key == 'T':
            Np = EXPS[exp][2]
            rng = np.random.default_rng([zlib.crc32(exp.encode()), N])
            return rng.gamma(np.maximum(d['M'], 1e-9), TAU / ((2 * N + 1) * Np))
    raise KeyError(key)


def series(exp, key, stat=np.mean):
    """Rows (N, mean, p5, p95) over every simulated N."""
    rows = []
    for N in pr.grid(exp):
        x = per_run(exp, N, key)
        x = x[np.isfinite(x)]
        if len(x):
            rows.append((N, stat(x), np.percentile(x, 5), np.percentile(x, 95)))
    return np.array(rows)


def fit(r, idx=None):
    """y = A N^b on the given rows (all by default). Returns (A, b)."""
    rr = r if idx is None else r[idx]
    b, a = np.polyfit(np.log(rr[:, 0]), np.log(rr[:, 1]), 1)
    return math.exp(a), b


def thirds(r):
    fs = [fit(c) for c in np.array_split(r, 3)]
    A = np.array([f[0] for f in fs]); b = np.array([f[1] for f in fs])
    return A.mean(), A.std(ddof=1), b.mean(), b.std(ddof=1)


def halves(r):
    k = len(r) // 2
    return fit(r[:k]), fit(r[k:])


def rng_txt(r, lo, hi):
    return f'{int(r[lo, 0]):,}–{int(r[hi, 0]):,}'


# ── drawing ───────────────────────────────────────────────────────────────────

def letter(ax, s):
    ax.text(-0.2, 1.06, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom')


def axes_log(ax, ylabel, logy=True):
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('network size')
    ax.set_ylabel(ylabel)


def dots(ax, r, color=BLUE, label=None):
    yerr = [np.clip(r[:, 1] - r[:, 2], 0, None), np.clip(r[:, 3] - r[:, 1], 0, None)]
    ax.errorbar(r[:, 0], r[:, 1], yerr=yerr, fmt='o', color=color, ms=3.5, elinewidth=0.9,
                capsize=2.2, capthick=0.9, label=label)


def fitline(ax, r, idx=None, label=None):
    A, b = fit(r, idx)
    rr = r if idx is None else r[idx]
    x = np.logspace(np.log10(rr[0, 0]), np.log10(rr[-1, 0]), 50)
    ax.plot(x, A * x ** b, color=RED, lw=0.9, label=label, zorder=5)
    return A, b


def image(ax, name):
    ax.imshow(plt.imread(os.path.join(HERE, 'panels', f'{name}.png')))
    ax.axis('off')


def xlim(ax, r):
    ax.set_xlim(r[0, 0] / 1.4, r[-1, 0] * 1.4)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def fmt_fit(A, b):
    return f'y = {A:.3g}x^{b:.2f}'


# ── figures ───────────────────────────────────────────────────────────────────

def fig3():
    fig = plt.figure(figsize=(8.2, 7.2))
    gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.38)
    ax = fig.add_subplot(gs[0, 0]); image(ax, 'fig3A')

    ax = fig.add_subplot(gs[0, 1]); letter(ax, 'B')
    r = series('par_1e4', 'M')
    axes_log(ax, 'mean number of mutations\nto evolve AND')
    A, b = fitline(ax, r, label='power law fit')
    ax.plot(r[:, 0], r[:, 0], color=BLACK, lw=0.9, label='y=x line')
    dots(ax, r)
    ax.legend(loc='upper left', fontsize=8)
    xlim(ax, r)
    t = thirds(r)
    FITS.append(('3B', f'power-law fit to all {len(r)} data points: A = {A:.2g}, b = {b:.2f} '
                       f'(paper, 12 points: A = 9.8, b = 0.96). Three non-overlapping subsets: '
                       f'A = {t[0]:.1f} ± {t[1]:.1f}, b = {t[2]:.2f} ± {t[3]:.2f} '
                       f'(paper: A = 9.6 ± 3.7, b = 0.96 ± 0.11).'))

    ax = fig.add_subplot(gs[1, 0]); letter(ax, 'C')
    rl, rf = series('par_1e4', 'lost'), series('par_1e4', 'fix')
    axes_log(ax, 'mean number of  mutations\nto evolve AND')
    A, b = fitline(ax, rl)
    dots(ax, rl, BLUE, 'non-fixating  mutations')
    dots(ax, rf, ORANGE, 'fixating  mutations')
    ax.legend(loc='upper left', fontsize=8)
    xlim(ax, rl)
    FITS.append(('3C', f'best fit to non-fixating mutations: {fmt_fit(A, b)} (paper: y = 9.5x^0.96).'))

    ax = fig.add_subplot(gs[1, 1]); letter(ax, 'D')
    axes_log(ax, 'fraction of non-fixating mutations', logy=False)
    for key, c, lab in [('frac_ben', GREEN, 'beneficial'), ('frac_neu', BLUE, 'neutral'),
                        ('frac_del', ORANGE, 'deleterious')]:
        rr = series('par_1e4', key)
        dots(ax, rr, c, lab)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='center right', fontsize=8)
    xlim(ax, rr)
    save(fig, 'fig3.png')


def fig4():
    fig = plt.figure(figsize=(8.2, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.15], wspace=0.35, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0]); image(ax, 'fig4A')

    ax = fig.add_subplot(gs[0, 1]); letter(ax, 'B')
    r = series('par_1e4', 'ce')
    axes_log(ax, 'mean number of computing  edges', logy=False)
    dots(ax, r)
    ax.set_ylim(1, 10); ax.set_yticks(range(1, 11))
    xlim(ax, r)

    ax = fig.add_subplot(gs[1, 1]); letter(ax, 'C')
    r = series('par_1e4', 'fb')
    axes_log(ax, 'mean fraction of beneficial\nmutations per computing edge', logy=False)
    dots(ax, r)
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.1))
    xlim(ax, r)

    ax = fig.add_subplot(gs[2, 1]); image(ax, 'fig4D')
    save(fig, 'fig4.png')


def fig5():
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 1, 1], wspace=0.5, hspace=0.4)
    ax = fig.add_subplot(gs[:, 0]); image(ax, 'fig5A')

    ax = fig.add_subplot(gs[0, 1]); letter(ax, 'B')
    r = series('chain_1e4', 'ce')
    axes_log(ax, 'mean number of computing edges')
    A, b = fitline(ax, r)
    dots(ax, r)
    xlim(ax, r)
    (_, b1), (_, b2) = halves(r)
    FITS.append(('5B', f'best fit to the data: {fmt_fit(A, b)} (paper: y = 2x^1). Two non-overlapping '
                       f'halves: exponents {b1:.2f} and {b2:.2f} (paper: 0.99 and 1).'))

    ax = fig.add_subplot(gs[0, 2]); letter(ax, 'C')
    r = series('chain_1e4', 'fb')
    axes_log(ax, 'mean fraction of beneficial\nmutations per computing edge', logy=False)
    dots(ax, r)
    ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.1))
    xlim(ax, r)

    ax = fig.add_subplot(gs[1, 1]); image(ax, 'fig5D')

    ax = fig.add_subplot(gs[1, 2]); letter(ax, 'E')
    r = series('chain_1e4', 'M')
    axes_log(ax, 'mean number of mutations\nto evolve AND')
    A, b = fitline(ax, r)
    dots(ax, r)
    ax.set_ylim(1, None)
    xlim(ax, r)
    (_, b1), (_, b2) = halves(r)
    FITS.append(('5E', f'power-law fit: {fmt_fit(A, b)} (paper: y = 31x^0.07). Two non-overlapping '
                       f'halves: exponents {b1:.2f} and {b2:.2f} (paper: 0.07 and −0.09).'))
    save(fig, 'fig5.png')


def figS1():
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    r = series('rw', 'M')
    axes_log(ax, 'number  of  mutations  to  evolve AND')
    A, b = fitline(ax, r)
    dots(ax, r)
    ax.set_ylim(1, None)
    xlim(ax, r)
    t = thirds(r)
    FITS.append(('S1', f'best fit to y = Ax^b: A = {A:.3g}, b = {b:.2f} (paper: A = 18.7, b = 0.8). '
                       f'Three non-overlapping subsets: b = {t[2]:.2f} ± {t[3]:.2f} (paper: 0.8 ± 0.2).'))
    save(fig, 'figS1.png')


def figS2():
    fig = plt.figure(figsize=(10, 3.2))
    gs = fig.add_gridspec(1, 3, wspace=0.35)
    image(fig.add_subplot(gs[0, 0]), 'figS2A')
    image(fig.add_subplot(gs[0, 1]), 'figS2B')
    ax = fig.add_subplot(gs[0, 2]); letter(ax, 'C')
    r = series('par_1e4', 'M', stat=np.median)
    axes_log(ax, 'median number of mutations\nto evolve AND')
    A, b = fitline(ax, r)
    ax.plot(r[:, 0], r[:, 1], 'o', color=BLUE, ms=5)
    xlim(ax, r)
    FITS.append(('S2C', f'power-law fit to the median: {fmt_fit(A, b)} (paper: y = 7.8x^0.96).'))
    save(fig, 'figS2.png')


def figS3():
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw=dict(wspace=0.42))
    paper = {'andnot': ('107.8', '0.82', '0.64, 0.95'), 'or': ('64.6', '0.86', '0.78, 0.94'),
             'true': ('34.8', '0.91', '0.86, 0.94')}
    for ax, L, (tag, ttl) in zip(axs, 'ABC', [('andnot', 'Target function: I1 AND (NOT I2)'),
                                              ('or', 'Target  function:  OR'),
                                              ('true', 'Target phenotype: TRUE')]):
        letter(ax, L)
        r = series(f'{tag}_10', 'M')
        axes_log(ax, 'mean number of mutations to evolve')
        A, b = fitline(ax, r)
        dots(ax, r)
        ax.set_title(ttl, fontsize=8.5, fontweight='bold')
        ax.set_ylim(10 ** math.floor(math.log10(r[:, 2].min())), 10 ** math.ceil(math.log10(r[:, 3].max())))
        xlim(ax, r)
        b1, b2 = fit(r[:3])[1], fit(r[-4:])[1]
        pA, pb, pbr = paper[tag]
        FITS.append((f'S3{L}', f'{ttl.split(":")[1].strip()}: fitted to {fmt_fit(A, b)} [b: {b1:.2f}, {b2:.2f}] '
                               f'(first three, last four points: N = {rng_txt(r, 0, 2)} and '
                               f'{rng_txt(r, -4, -1)}; paper: y = {pA}x^{pb} [b: {pbr}]).'))
    save(fig, 'figS3.png')


def small_pop(name, exp, exp_big, paper):
    fig, axs = plt.subplots(2, 2, figsize=(7.6, 7.0), gridspec_kw=dict(wspace=0.45, hspace=0.4))
    chain = exp.startswith('chain')
    ax = axs[0, 0]; letter(ax, 'A')
    r = series(exp, 'M')
    axes_log(ax, 'mean number of mutations\nto evolve AND')
    A, b = fitline(ax, r)
    dots(ax, r)
    ax.set_ylim(1, None)
    if not chain:
        ax.set_title('Np = 10', fontsize=9, fontweight='bold')
    xlim(ax, r)
    (_, b1), (_, b2) = halves(r)
    k = len(r) // 2
    FITS.append((f'{name}A', f'power-law fit to all data points: {fmt_fit(A, b)}; two non-overlapping '
                             f'subsets (N = {rng_txt(r, 0, k - 1)}, {rng_txt(r, k, -1)}): '
                             f'{b1:.2f} and {b2:.2f} ({paper["A"]}).'))

    ax = axs[0, 1]; letter(ax, 'B')
    r10, rbig = series(exp, 'fix'), series(exp_big, 'fix')
    axes_log(ax, 'Mean number of fixating mutations\nto evolve AND')
    dots(ax, r10, GREEN, '$\\mathregular{N_p}$ = 10')
    dots(ax, rbig, ORANGE, '$\\mathregular{N_p}$ = $\\mathregular{10^4}$')
    ax.set_ylim(1, None)
    ax.legend(loc='upper left', fontsize=8)
    xlim(ax, rbig)

    ax = axs[1, 0]; letter(ax, 'C')
    r = series(exp, 'pc')
    axes_log(ax, 'probability of choosing\na computing edge', logy=not chain)
    A, b = fitline(ax, r)
    dots(ax, r)
    if chain:
        ax.set_ylim(0, 1.1)
    xlim(ax, r)
    (_, b1), (_, b2) = halves(r)
    FITS.append((f'{name}C', f'P(c): power-law fit to all data points, exponent {b:.3f}; two '
                             f'non-overlapping subsets (N = {rng_txt(r, 0, k - 1)}, {rng_txt(r, k, -1)}): '
                             f'{b1:.3f} and {b2:.3f} ({paper["C"]}).'))

    ax = axs[1, 1]; letter(ax, 'D')
    r = series(exp, 'fb')
    axes_log(ax, 'mean fraction of beneficial\nmutations per computing edge', logy=False)
    A, b = fitline(ax, r)
    dots(ax, r)
    ax.set_ylim(0, 1)
    xlim(ax, r)
    (_, b1), (_, b2) = halves(r)
    FITS.append((f'{name}D', f'P(b|c): power-law fit to all data points, exponent {b:.3f}; two '
                             f'non-overlapping subsets (N = {rng_txt(r, 0, k - 1)}, {rng_txt(r, k, -1)}): '
                             f'{b1:.3f} and {b2:.3f} ({paper["D"]}).'))
    save(fig, f'fig{name}.png')


def figS6():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(wspace=0.4))
    ax = axs[0]; letter(ax, 'A')
    r = series('par_1e4', 'T')
    axes_log(ax, 'time  to  evolve AND', logy=False)
    dots(ax, r)
    ax.set_ylim(0, 2); ax.set_yticks(np.arange(0, 2.01, 0.2))
    xlim(ax, r)
    ax = axs[1]; letter(ax, 'B')
    r = series('chain_1e4', 'T')
    axes_log(ax, 'time  to  evolve AND')
    dots(ax, r)
    xlim(ax, r)
    save(fig, 'figS6.png')


def write_fits():
    rng = 'N ≤ 10⁴' if TAG == '1e4' else 'every simulated N (up to 10⁶)'
    lines = [f'# Caption numbers for the extended figures ({TAG} set)', '',
             f'Computed exactly as the paper\'s captions describe, on {rng}. '
             'The paper\'s values (N ≤ 10³) are in brackets. Generated by `figures.py`.', '']
    lines += [f'- **Fig. {f}:** {t}' for f, t in FITS]
    out = os.path.join(HERE, f'fits_{TAG}.md')
    with open(out, 'w') as fh:
        fh.write('\n'.join(lines).replace('-0.', '−0.').replace('-1.', '−1.') + '\n')
    print('wrote', out)


def run(tag):
    global OUT, TAG
    TAG = tag
    OUT = os.path.join(HERE, f'figs_{tag}')
    pr.NMAX = int(float(tag))
    pr._CACHE.clear() if hasattr(pr, '_CACHE') else None
    FITS.clear()
    fig3(); fig4(); fig5(); figS1(); figS2(); figS3()
    small_pop('S4', 'par_10', 'par_1e4', dict(
        A='paper: 0.90; 0.87 and 0.92 for 5–40 and 60–600',
        C='paper: −0.85; −0.81 and −0.87 for 5–40 and 60–600',
        D='paper: −0.020; −0.049 and −0.012 for 5–40 and 60–600'))
    small_pop('S5', 'chain_10', 'chain_1e4', dict(
        A='paper: y = 41.3x^0.10; 0.21 and −0.17 for 5–41 and 81–501',
        C='paper: 0.027; 0.046 and 0.014 for 5–41 and 81–501',
        D='paper: −0.135; −0.238 and −0.036 for 5–41 and 81–501'))
    figS6()
    write_fits()


def main():
    import sys
    for tag in (sys.argv[1:] or ['1e4', '1e6']):
        run(tag)


if __name__ == '__main__':
    main()
