"""
The paper's figures (arXiv 2610.40635), redrawn from faster_circuit runs, extended
to large N and corrected where our understanding has moved on (see CHANGES.md).

  python -m paper_replication.figures 1e4    ->  paper_replication/figs_1e4/
  python -m paper_replication.figures 1e6    ->  paper_replication/figs_1e6/

The 1e4 set shows N <= 10^4; the 1e6 set shows everything simulated.

Estimators (one policy for every figure):
  * per-run counts (M, fixations, losses, beneficial proposals, ...): mean over
    runs, thick bar = 95% run-bootstrap CI of the plotted statistic; on M a thin
    bar also shows the 5-95% spread of the runs. Where a few runs dominate the
    total (chain start), the median is the primary statistic and the mean is
    drawn hollow.
  * every ratio (P(c), P(b|c), P_fix, loss fractions, computing edges, f_b) is a
    ratio of sums over runs ("pooled"), with a run-bootstrap CI. Pooled ratios
    are what the exact theory predicts and the only ones for which Eq. 2 is an
    identity. Per-run medians are added hollow where runs are very unequal.
  * references are exact theory (bottleneck_replication.theory), not fits.
"""
import os
import sys
import math
import zlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_replication.run import EXPS, path, AND, ANDNOT, OR, TRUE, FB_DEF
from bottleneck_replication.theory import (exact_M_parallel, exact_M_parallel_terms, inv_p0_formula,
                                           p_chain_reservoir, enumerate_p, _ab)

HERE = os.path.dirname(__file__)
INK, MUTED, GRID_C = '#1a1a1a', '#6b6b6b', '#e6e6e6'
BLUE, ORANGE, GREEN, PINK, SKY, VERM, YEL, PURP = ('#0072B2', '#E69F00', '#009E73', '#CC79A7',
                                                  '#56B4E9', '#D55E00', '#F0E442', '#2E2A7A')
GREY = '#9a9a9a'
# fitness categories, viridis as in the paper's Figs 2, 4D, 5D
FITC = {0.0: '#440154', 0.25: '#3b528b', 0.5: '#21918c', 0.75: '#5ec962', 1.0: '#fde725'}
TAU = 1000.0                     # 1 / mutation rate per edge per network (Methods 6.1)
NBOOT = 1000

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333', 'axes.linewidth': 0.9, 'legend.frameon': False,
                     'savefig.dpi': 150, 'axes.titlesize': 12.5})

NMAX = 10**4
PI_H = _ab(1e4)[0]               # pi(1/2) at N_P = 10^4


# ── data ──────────────────────────────────────────────────────────────────────

_CACHE = {}


def load(exp, N):
    """Per-run arrays for one (experiment, N). Column meaning: observe.STAT_COLS."""
    if (exp, N) in _CACHE:
        return _CACHE[exp, N]
    p = path(exp, N)
    if not os.path.exists(p):
        _CACHE[exp, N] = None
        return None
    d = np.load(p)
    st = d['stats'].astype(float)
    n = len(st)
    nan = np.full(n, np.nan)
    out = dict(M=st[:, 0], fix_ben=st[:, 1], fix_del=st[:, 2], fix_neu=st[:, 3],
               lost_ben=st[:, 4], lost_del=st[:, 5], lost_neu=st[:, 6],
               n_res=st[:, 7], ce_sum=st[:, 8], fb_sum=st[:, 9], fb_n=st[:, 10],
               capped=st[:, 11], counts=d['counts'].astype(float))
    out['fix'] = st[:, 1:4].sum(1)
    out['lost'] = st[:, 4:7].sum(1)
    out['n_comp'] = st[:, 12] if st.shape[1] > 12 else nan       # older files: no counter
    out['n_ben'] = out['fix_ben'] + out['lost_ben']
    fb_def = str(d['fb_def']) if 'fb_def' in d.files else ''
    if fb_def != FB_DEF or not st[:, 10].any():
        out['fb_sum'] = out['fb_n'] = nan                          # never plot old-convention f_b
    # f_b is measured on each resident with probability min(1, r_fb / n_res):
    # weight runs by the inverse to pool over all residents
    r_fb = float(d['r_fb']) if 'r_fb' in d.files else 0.0
    out['w_fb'] = np.maximum(1.0, out['n_res'] / r_fb) if r_fb > 0 else np.ones(n)
    with np.errstate(invalid='ignore', divide='ignore'):
        out['pc_mut'] = out['n_comp'] / out['M']
        out['pbc_mut'] = out['n_ben'] / out['n_comp']
    c = out['counts']
    out['M0'] = c[:, 0] + c[:, 1]          # proposals below fitness 1/2 (Eq. 3, first term)
    out['Mh'] = c[:, 2]                    # proposals at fitness 1/2 (second term)
    out['Mn'] = out['M'] / N
    out['Tn'] = out['M'] / (2 * N + 1)     # E[T | M] in units of tau / N_P (Methods 6.1)
    out['N'] = N
    _CACHE[exp, N] = out
    return out


def grid(exp, nmax=None):
    nmax = nmax or NMAX
    return [N for N in EXPS[exp][3] if N <= nmax * 1.0001 + 1 and load(exp, N) is not None]


def _rng(*key):
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def series(exp, key, nmax=None, stat='mean', scale=None):
    """Rows (N, centre, p5, p95, ci_lo, ci_hi, n): centre = mean or median of the
    per-run values; p5/p95 = spread of the runs; ci = 95% run-bootstrap CI."""
    rows = []
    for N in grid(exp, nmax):
        x = load(exp, N)[key]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        if scale is not None:
            x = x * scale(N)
        f = np.median if stat == 'median' else np.mean
        idx = _rng(exp, N, key, stat).integers(0, len(x), (NBOOT, len(x)))
        bs = f(x[idx], axis=1)
        rows.append((N, f(x), np.percentile(x, 5), np.percentile(x, 95),
                     np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(x)))
    return np.array(rows) if rows else np.zeros((0, 7))


def pooled(exp, num, den, nmax=None, w=None, scale=None):
    """Rows (N, sum(num)/sum(den), nan, nan, ci_lo, ci_hi, n) over runs, with a
    run-bootstrap CI. w: per-run weight key. Rows with missing data are skipped."""
    rows = []
    for N in grid(exp, nmax):
        d = load(exp, N)
        a, b = d[num], d[den]
        ww = d[w] if w else np.ones(len(a))
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or b.sum() <= 0:
            continue
        a, b = a * ww, b * ww
        idx = _rng(exp, N, num, den).integers(0, len(a), (NBOOT, len(a)))
        with np.errstate(invalid='ignore', divide='ignore'):
            bs = a[idx].sum(1) / b[idx].sum(1)
        k = scale(N) if scale else 1.0
        rows.append((N, k * a.sum() / b.sum(), np.nan, np.nan,
                     k * np.nanpercentile(bs, 2.5), k * np.nanpercentile(bs, 97.5), len(a)))
    return np.array(rows) if rows else np.zeros((0, 7))


def top_share(exp, N, q=0.01):
    """Fraction of all proposals (sum of M) made by the longest q of the runs."""
    M = np.sort(load(exp, N)['M'])[::-1]
    k = max(1, int(round(q * len(M))))
    return M[:k].sum() / M.sum()


# ── exact theory for the parallel start at N_P = 10^4 (two-step path) ─────────

def par_theory(N, Np=1e4):
    """Two-step path 0 -> 1/2 -> 1 (bottleneck note, Eq. 3). Step 1: 3 computing
    edges, 2 of them beneficial for every new source. Step 2: 5 computing edges,
    1 of them beneficial for N-1 of N sources. Neutral fixations add residents at
    rate 1/N_P per proposal (they land in the reservoir). Returns pooled limits."""
    N = np.asarray(N, float)
    t1, t2 = exact_M_parallel_terms(N, Np)
    a, b = 1 + t1 / Np, 1 + t2 / Np                       # residents at each step
    q = (N - 1) / N
    n_comp = (3 * t1 + 5 * t2) / (2 * N + 1)
    n_ben = (2 * t1 + q * t2) / (2 * N + 1)
    return dict(M=t1 + t2, t1=t1, t2=t2,
                # per resident, final included: 7 computing edges, or 5 if the bottleneck
                # rewiring picks gate 2 itself (probability 1/(N-1))
                ce=(3 * a + 5 * b + 7 - 2 / (N - 1)) / (a + b + 1),
                fb=(2 * a + q * b) / (3 * a + 5 * b),        # per edge, final resident excluded
                pc2=(3 * t1 + 5 * t2) / (t1 + t2),           # (2N+1) P(c), per proposal
                pc=n_comp / (t1 + t2), pbc=n_ben / n_comp, pfix=2 / n_ben, pb=2 / (t1 + t2),
                n_ben=n_ben)


def hypo_survival(x, m1, m2):
    """P(X > x) for X = sum of two exponentials with means m1, m2 (x in absolute units)."""
    if abs(m1 - m2) < 1e-9 * m1:
        return (1 + x / m1) * np.exp(-x / m1)
    return (m2 * np.exp(-x / m2) - m1 * np.exp(-x / m1)) / (m2 - m1)


def hypo_median(m1, m2):
    lo, hi = 0.0, 20 * (m1 + m2)
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if hypo_survival(mid, m1, m2) > 0.5 else (lo, mid)
    return (lo + hi) / 2


def geom_sum_pmf(m1, m2, xmax):
    """P(G1 + G2 = x), G_i geometric on {1, 2, ...} with means m_i, for x < xmax."""
    k = np.arange(xmax)
    g1 = np.where(k >= 1, (1 / m1) * (1 - 1 / m1) ** np.maximum(k - 1, 0), 0)
    g2 = np.where(k >= 1, (1 / m2) * (1 - 1 / m2) ** np.maximum(k - 1, 0), 0)
    return np.convolve(g1, g2)[:xmax]


def wls_exponent(r, lo=0, hi=np.inf):
    """b +- se of log y = a + b log N, weighted by the CI half-width, se inflated
    by sqrt(chi2/dof). For the README; no fits are drawn."""
    m = (r[:, 0] >= lo) & (r[:, 0] <= hi) & (r[:, 1] > 0)
    if m.sum() < 3:
        return None
    x, y = np.log(r[m, 0]), np.log(r[m, 1])
    s = np.maximum((np.log(r[m, 5]) - np.log(np.maximum(r[m, 4], 1e-300))) / 3.92, 1e-6)
    W = 1 / s ** 2
    X = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(X.T @ (W[:, None] * X))
    beta = cov @ X.T @ (W * y)
    chi2 = (W * (y - X @ beta) ** 2).sum() / max(m.sum() - 2, 1)
    return beta[1], math.sqrt(cov[1, 1] * max(chi2, 1.0))


# ── drawing helpers ───────────────────────────────────────────────────────────

def title(ax, s):
    ax.set_title(s, loc='left', fontweight='bold')


def logx(ax, ylabel=None, xlabel='network size $N$'):
    ax.set_xscale('log'); ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which='major', color=GRID_C, lw=0.6, zorder=0)


def loglog(ax, ylabel=None, xlabel='network size $N$'):
    logx(ax, ylabel, xlabel); ax.set_yscale('log')


def dots(ax, r, color, label=None, marker='o', hollow=False, offset=1.0, ms=6.5,
         spread=False, ci=True):
    """Points at r[:,1]; thick bar = CI (r[:,4:6]); thin bar = 5-95% spread (r[:,2:4])."""
    if not len(r):
        return
    x = r[:, 0] * offset
    if spread:
        ax.errorbar(x, r[:, 1], yerr=[np.clip(r[:, 1] - r[:, 2], 0, None), np.clip(r[:, 3] - r[:, 1], 0, None)],
                    fmt='none', ecolor=color, elinewidth=0.9, alpha=0.35, capsize=0, zorder=2)
    if ci:
        ax.errorbar(x, r[:, 1], yerr=[np.clip(r[:, 1] - r[:, 4], 0, None), np.clip(r[:, 5] - r[:, 1], 0, None)],
                    fmt='none', ecolor=color, elinewidth=2.4, alpha=0.9, capsize=0, zorder=2)
    kw = dict(mfc='white', mec=color, mew=1.6) if hollow else dict(mfc=color, mec=INK, mew=0.6)
    ax.plot(x, r[:, 1], marker, ms=ms, ls='none', label=label, zorder=3, **kw)


def curve(ax, xs, f, color, label=None, ls='-', lw=1.3, alpha=1.0):
    xx = np.logspace(np.log10(xs[0]), np.log10(xs[-1]), 200)
    ax.plot(xx, f(xx), color=color, ls=ls, lw=lw, label=label, alpha=alpha, zorder=1)


def note(ax, s, loc=(0.98, 0.04), ha='right', va='bottom', fs=9):
    ax.text(*loc, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=MUTED)


def num(x, fmt='{:.2f}'):
    """Number with a typographic minus sign."""
    return fmt.format(x).replace('-', '−')


def _limits(ax):
    """Log-axis limits from all plotted data (lines and error bars), with margin."""
    from matplotlib.collections import LineCollection
    xs, ys = [], []
    for ln in ax.get_lines():
        if ln.get_label() == '_ref' or ln.get_transform() != ax.transData:
            continue                               # axhline/axvline: not data
        xs.append(np.asarray(ln.get_xdata(), float)); ys.append(np.asarray(ln.get_ydata(), float))
    for c in ax.collections:
        if isinstance(c, LineCollection):
            for seg in c.get_segments():
                xs.append(seg[:, 0]); ys.append(seg[:, 1])
    if not xs:
        return
    x = np.concatenate(xs); y = np.concatenate(ys)
    if ax.get_xscale() == 'log' and not getattr(ax, 'keep_xlim', False):
        x = x[np.isfinite(x) & (x > 0)]
        if len(x):
            ax.set_xlim(x.min() / 1.6, x.max() * 1.6)
    if ax.get_yscale() == 'log' and not getattr(ax, 'keep_ylim', False):
        y = y[np.isfinite(y) & (y > 0)]
        if len(y):
            ax.set_ylim(y.min() / 1.6, y.max() * 2.2)


def save(fig, name, outdir):
    for ax in fig.axes:
        if ax.axison:
            _limits(ax)
    fig.tight_layout()
    p = os.path.join(outdir, name)
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


# ── network drawings ──────────────────────────────────────────────────────────

def node(ax, xy, label, fill='white', w=0.9, h=0.55, fs=10, ec=INK):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=fill, ec=ec, lw=1.2, zorder=3))
    ax.text(*xy, label, ha='center', va='center', fontsize=fs, zorder=4)


def arrow(ax, a, b, color=INK, lw=1.3, ls='-', rad=0.0, h=0.55):
    """Edge from node centre a (bottom side) to node centre b (top side)."""
    p = FancyArrowPatch((a[0], a[1] - h / 2), (b[0], b[1] + h / 2),
                        connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>', mutation_scale=11,
                        color=color, lw=lw, ls=ls, zorder=2, shrinkA=1, shrinkB=1)
    ax.add_patch(p)


def blank(ax, xlim=(0, 10), ylim=(0, 10)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')


def draw_parallel(ax, n_show=4, computing=None, rewired=False):
    """Parallel start (Fig. 3A). computing: colour for computing edges."""
    I1, I2 = (3.2, 8.6), (6.8, 8.6)
    gx = np.linspace(1.4, 8.6, n_show)
    gates = [(x, 5.6) for x in gx]
    O = (gx[0], 2.6)
    node(ax, I1, '$I_1$', '#e3e3e3'); node(ax, I2, '$I_2$', '#e3e3e3')
    for i, g in enumerate(gates):
        node(ax, g, str(i + 1) if i < n_show - 1 else '$N$')
    ax.text((gx[-2] + gx[-1]) / 2, 5.6, '…', ha='center', va='center', fontsize=14)
    node(ax, O, '$O$', '#e3e3e3')
    cc = computing or INK
    for i, g in enumerate(gates):
        c = cc if i == 0 else '#9a9a9a'
        if i == 0 and rewired:
            arrow(ax, gates[1], g, color=GREEN, lw=2, ls='--', rad=0.0, h=0)
            arrow(ax, I2, g, color=cc, lw=2 if computing else 1.2)
        else:
            arrow(ax, I1, g, color=c, lw=2 if (computing and i == 0) else 1.0)
            arrow(ax, I2, g, color=c, lw=2 if (computing and i == 0) else 1.0)
    if rewired:
        for s in (I1, I2):
            arrow(ax, s, gates[1], color=cc, lw=2)
    arrow(ax, gates[0], O, color=cc, lw=2 if computing else 1.3)


def draw_chain(ax, n_show=4):
    I1, I2 = (2.0, 9.0), (4.6, 9.0)
    ys = [7.5, 6.2, 4.9]
    x = 3.3
    node(ax, I1, '$I_1$', '#e3e3e3'); node(ax, I2, '$I_2$', '#e3e3e3')
    for i, y in enumerate(ys):
        node(ax, (x, y), str(i + 1))
    node(ax, (x, 2.9), '$N$')
    node(ax, (x, 1.3), '$O$', '#e3e3e3')
    arrow(ax, I1, (x, ys[0]), color=VERM, lw=1.8)
    arrow(ax, I2, (x, ys[0]), color=VERM, lw=1.8)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x - 0.15, y0), (x - 0.15, y1), color=VERM, lw=1.8)
        arrow(ax, (x + 0.15, y0), (x + 0.15, y1), color=VERM, lw=1.8)
    ax.plot([x - 0.15, x - 0.15], [ys[-1] - 0.35, 3.25], color=VERM, lw=1.8, ls=':')
    ax.plot([x + 0.15, x + 0.15], [ys[-1] - 0.35, 3.25], color=VERM, lw=1.8, ls=':')
    arrow(ax, (x, 2.9), (x, 1.3), color=VERM, lw=1.8)


# ── figure 1: model ───────────────────────────────────────────────────────────

def tt_str(v):
    return [str((v >> i) & 1) for i in range(4)]


I1_TT, I2_TT = 0b1100, 0b1010


def nand(a, b):
    return (~(a & b)) & 0xF


def fit_and(v):
    return sum(o == t for o, t in zip(tt_str(v), tt_str(AND))) / 4


def box(ax, x0, y0, x1, y1, text, fc='#f4f7fb', fs=9.6):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc=fc, ec='#8aa1b8', lw=1.1, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs, zorder=3)


def poly_arrow(ax, pts, color=INK, label=None, lpos=None, lw=1.1):
    """Orthogonal connector through pts, arrowhead on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_joinstyle='round', zorder=1)
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=9, ha='center', va='bottom', color=color)


def two_gate_net(ax, ox, edges, fitness=None):
    """Fig. 1 two-gate network at x offset ox. edges: (src, dst, color, ls, dx) with
    nodes 'I1', 'I2', '1', '2', 'O'; dx shifts a parallel copy of an edge."""
    P = {'I1': (ox + 1.4, 8.8), 'I2': (ox + 4.4, 8.8), '2': (ox + 3.9, 6.5), '1': (ox + 1.9, 4.2),
         'O': (ox + 1.9, 1.9)}
    for k, lab in [('I1', '$I_1$'), ('I2', '$I_2$'), ('O', '$O$')]:
        node(ax, P[k], lab, '#e3e3e3')
    node(ax, P['1'], '1'); node(ax, P['2'], '2')
    for a, b, c, ls, dx in edges:
        arrow(ax, (P[a][0] + dx, P[a][1]), (P[b][0] + dx, P[b][1]), color=c, ls=ls,
              lw=2.0 if c != INK else 1.3)
    if fitness is not None:
        ax.text(P['O'][0], 0.8, f'fitness = {fitness:g}', ha='center', fontsize=10)
    return P


def fig1(outdir):
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.95], width_ratios=[1, 1.05])

    # A: gate 2 = NAND(I1, I2), gate 1 = NAND(I1, gate 2), O <- gate 1 (paper Fig. 1A)
    ax = fig.add_subplot(gs[0, 0]); blank(ax, (0, 14), (0, 10))
    title(ax, 'A   network (genotype) and its function (phenotype)')
    two_gate_net(ax, 0, [('I1', '2', INK, '-', 0), ('I2', '2', INK, '-', 0),
                         ('I1', '1', INK, '-', 0), ('2', '1', INK, '-', 0), ('1', 'O', INK, '-', 0)])
    v2 = nand(I1_TT, I2_TT); v1 = nand(I1_TT, v2)
    rows = [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]
    out, tgt = tt_str(v1), tt_str(AND)
    tx, ty = 7.2, 7.6
    for j, h in enumerate(['$I_1$', '$I_2$', 'output', 'AND']):
        ax.text(tx + 1.25 * j, ty + 0.9, h, fontsize=10, fontweight='bold', ha='center')
    for i, (a, b) in enumerate(rows):
        ok = out[i] == tgt[i]
        for j, v in enumerate([a, b, out[i], tgt[i]]):
            ax.text(tx + 1.25 * j, ty - 0.8 * i, v, fontsize=10, family='monospace', ha='center')
        ax.text(tx + 4.85, ty - 0.8 * i, '✓' if ok else '✗', color=GREEN if ok else VERM,
                fontsize=11, ha='center')
    ax.text(tx - 0.4, ty - 3.9, f'fitness = {sum(o == t for o, t in zip(out, tgt))}/4 = {fit_and(v1):g}',
            fontsize=10.5)
    ax.text(0.2, 0.0, 'every gate is NAND; network size $N$ = number of gates', fontsize=9, color=MUTED)

    # B: the paper's mutation: one of the two I1 -> 1 edges is replaced by 2 -> 1
    ax = fig.add_subplot(gs[1, :]); blank(ax, (0, 32), (0, 10))
    title(ax, 'B   a mutation removes one edge and rewires it to a new source')
    base = [('I1', '2', INK, '-', 0), ('I2', '2', INK, '-', 0), ('1', 'O', INK, '-', 0)]
    f0 = fit_and(nand(I1_TT, I1_TT)); f1 = fit_and(v1)
    for ox in (0, 11.5):
        P = two_gate_net(ax, ox, base + [('I1', '1', INK, '-', -0.14),
                                         ('I1', '1', VERM if ox else INK, '-', 0.14)],
                         fitness=None if ox else f0)
        ax.plot([P['2'][0]] * 2, [P['2'][1] - 0.28, P['2'][1] - 0.75], color=INK, lw=1.3)  # unused output
    P = two_gate_net(ax, 23, base + [('I1', '1', INK, '-', 0), ('2', '1', GREEN, '-', 0)], fitness=f1)
    ax.annotate('', xy=(10.4, 5.6), xytext=(6.4, 5.6), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
    ax.text(8.4, 6.0, 'pick one of the $2N+1$ edges, uniformly', ha='center', fontsize=10)
    ax.text(8.4, 4.9, '$I_1 \\to 1$ (red) is removed', ha='center', fontsize=9.5, color=VERM)
    ax.annotate('', xy=(21.9, 5.6), xytext=(17.9, 5.6), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
    ax.text(19.9, 6.0, 'pick a new source, uniformly', ha='center', fontsize=10)
    ax.text(19.9, 4.9, 'choices for new edge:\n$I_2 \\to 1$,  $2 \\to 1$ (green)', ha='center',
            va='top', fontsize=9.5)
    ax.text(0.2, -0.25, 'new source: one of the $N$ nodes other than the old source and the gate itself (inputs '
                      'included; it may be the gate\'s other input, giving NAND$(x,x)$ = NOT $x$); for the edge into $O$: '
                      'one of the $N-1$ other gates.\nA proposal that would close a cycle is discarded and the whole '
                      'draw (edge and source) is repeated, so edges whose alternatives mostly close cycles are '
                      'mutated less often; discarded draws are not counted.', fontsize=9, color=MUTED, va='top')

    # C: flowchart (paper Fig. 1C)
    ax = fig.add_subplot(gs[0, 1]); blank(ax)
    title(ax, 'C   evolutionary algorithm (SSWM, Moran fixation)')
    box(ax, 0.2, 8.4, 4.4, 9.5, 'initialize resident network\nof size $N$')
    box(ax, 0.2, 6.0, 4.4, 7.3, 'propose an acyclic mutation &\ncompute the mutant\'s fitness $f\'$')
    box(ax, 0.2, 2.9, 4.4, 4.9, 'does the mutation fixate?\n'
                                '$\\pi(s) = \\frac{1-e^{-s}}{1-e^{-N_P s}}$\n'
                                '$s = f\' - f$, $N_P$ = population size', fs=9.2)
    box(ax, 5.4, 4.8, 9.0, 5.8, 'keep resident\ngenotype as it is')
    box(ax, 5.4, 2.0, 9.0, 3.0, 'update resident to\nthe mutated genotype')
    box(ax, 6.2, 0.2, 8.2, 0.95, 'end', fc='#fdf3e3')
    poly_arrow(ax, [(2.3, 8.4), (2.3, 7.3)])
    poly_arrow(ax, [(2.3, 6.0), (2.3, 4.9)])
    poly_arrow(ax, [(4.4, 3.9), (4.9, 3.9), (4.9, 5.3), (5.4, 5.3)], label='no', lpos=(4.65, 4.95))
    poly_arrow(ax, [(4.4, 3.9), (4.9, 3.9), (4.9, 2.5), (5.4, 2.5)], label='yes', lpos=(4.65, 2.6))
    poly_arrow(ax, [(9.0, 5.3), (9.35, 5.3), (9.35, 6.45), (4.4, 6.45)])
    poly_arrow(ax, [(9.0, 2.5), (9.75, 2.5), (9.75, 6.85), (4.4, 6.85)])
    ax.text(9.9, 4.0, 'if $f < 1$', fontsize=9, rotation=90, ha='left', va='center')
    poly_arrow(ax, [(7.2, 2.0), (7.2, 0.95)], label='', lpos=None)
    ax.text(7.35, 1.3, 'if $f = 1$', fontsize=9, ha='left', va='center')
    ax.text(2.45, 5.45, 'a proposal that closes a cycle is redrawn\n(edge and source) and not counted',
            fontsize=8.2, color=MUTED, va='center')
    ax.text(0.2, -0.3, '$f$: resident fitness. $M$ counts every acyclic proposal, fixed or lost (including neutral '
                       'ones\noutside the computing module); redrawn cyclic proposals are not counted. '
                       '$\\pi(s)$ is the Moran fixation\nprobability for relative fitness $e^{s}$ '
                       '($1/N_P$ at $s = 0$).', fontsize=9, color=MUTED, va='top')
    save(fig, 'fig1_model.png', outdir)


# ── figure 2: genotype graph of size-2 networks ───────────────────────────────

def fig2(outdir):
    """Genotype graph from the repo's gen_fig2.py (unordered input pairs, as in the
    paper: 54 genotypes), restyled."""
    import networkx as nx
    from matplotlib.patches import Patch
    import gen_fig2 as g2                              # graph construction only
    from faster_circuit.core import fitness, has_cycle_via
    from goals import and_funct
    # canonical genotypes as in gen_fig2.py (unordered input pairs), but fitness and
    # acyclicity from faster_circuit
    p0 = sorted((a, b) for a in range(4) for b in range(a, 4) if 2 not in (a, b))
    p1 = sorted((a, b) for a in range(4) for b in range(a, 4) if 3 not in (a, b))
    outs = [2, 3]

    def acyclic(a, b):
        g = [a[0], a[1], 0, 0, 2]
        return not any(has_cycle_via(g, 3, u) for u in b)

    genotypes = [(a, b, o) for a in p0 for b in p1 for o in outs if acyclic(a, b)]
    fit_ = {g: fitness([*g[0], *g[1], g[2]], and_funct) for g in genotypes}
    G = g2.build_graph(genotypes, p0, p1, outs)
    pos = nx.spring_layout(G, seed=7, k=1.8, iterations=300)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#9a9a9a', width=0.8, alpha=0.6)
    handles = []
    for f, c in FITC.items():
        nodes = [g for g in G if fit_[g] == f]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=c, node_size=230,
                               edgecolors=INK, linewidths=0.6, ax=ax)
        handles.append(Patch(fc=c, ec=INK, lw=0.6, label=f'{f:g}   ({len(nodes)} genotypes)'))
    start = ((0, 1), (0, 1), 2)                     # both gates NAND(I1, I2), O <- gate 1
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='none', node_size=520,
                           edgecolors=VERM, linewidths=2.2, ax=ax)
    ax.annotate('parallel start\n(NAND, fitness 0)', pos[start], xytext=(1.02, 0.30),
                textcoords='axes fraction', fontsize=9.5, color=VERM, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=VERM, lw=1, alpha=0.8))
    ax.axis('off')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=9.5,
              title='fitness for AND', title_fontsize=10)
    note(ax, 'genotype = (unordered input pair of gate 1, of gate 2, output gate); gates are labelled; '
             'NAND$(x,x)$ allowed;\nnon-computing gates included. Links = one rewired edge or a switch of the output '
             'gate. Links are not\nequally likely: from the parallel start the output switch has probability 0.2, '
             'each other link 0.1.', loc=(0.0, -0.02), ha='left', va='top')
    title(ax, f'Genotype graph of all acyclic networks with $N = 2$ '
              f'({G.number_of_nodes()} genotypes, {G.number_of_edges()} single-mutation links)')
    save(fig, 'fig2_genotype_graph.png', outdir)


# ── figure 3: NAND -> AND from the parallel start, N_P = 10^4 ─────────────────

def n_runs(exp):
    ns = {len(load(exp, N)['M']) for N in grid(exp)}
    return '/'.join(str(n) for n in sorted(ns))


def fig3(outdir, big):
    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 6)
    axA, axB, axE = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    axC, axD = fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])
    Ns = grid('par_1e4')

    blank(axA)
    draw_parallel(axA, computing=SKY)
    title(axA, 'A   starting network: $N$ parallel NAND$(I_1, I_2)$')
    note(axA, 'output reads gate 1: computes NAND, fitness 0 for AND\n'
              f'computing edges (blue): 3 of the $2N+1$\n'
              f'$N_P = 10^4$; {n_runs("par_1e4")} runs per $N$', loc=(0.02, 0.0), ha='left')

    ax = axB; loglog(ax, ylabel='total mutations proposed, $M$ (mean)')
    r = series('par_1e4', 'M')
    curve(ax, (4, Ns[-1] * 1.3), lambda n: 3 * n / PI_H, MUTED, ls=':', lw=1.2,
          label='asymptote $3N/\\pi(\\frac{1}{2}) = 7.62\\,N$')
    curve(ax, (4, Ns[-1] * 1.3), exact_M_parallel, PINK, lw=1.6,
          label='exact $\\langle M \\rangle$ (bottleneck note, Eq. 3)')
    dots(ax, r, BLUE, label='simulation: mean, 95% CI (thick), 5–95% of runs (thin)', spread=True)
    paper_grid = np.array([n for n in Ns if n <= 1000], float)
    bx = np.polyfit(np.log(paper_grid), np.log(exact_M_parallel(paper_grid)), 1)[0]
    ax.legend(loc='upper left', fontsize=9)
    note(ax, f'paper: $9.8\\,N^{{0.96}}$ fitted on $N \\leq 10^3$; the exact $\\langle M \\rangle$\n'
             f'fitted on the same $N$ gives $N^{{{bx:.2f}}}$: curvature, the slope tends to 1')
    title(ax, 'B   $\\langle M \\rangle$ follows exact theory and becomes linear')

    ax = axE; logx(ax, ylabel='mutations proposed / $N$ (mean)')
    th = lambda n, k: par_theory(n)[k] / n
    for key, k, c, lab, mk in [('Mn', 'M', BLUE, 'all ($M$)', 'o'),
                               ('Mh', 't2', VERM, 'at fitness ½: second step', 's'),
                               ('M0', 't1', GREEN, 'below fitness ½: first step', 'D')]:
        if key == 'Mn':
            rr = series('par_1e4', key)
        else:
            rr = series('par_1e4', key, scale=lambda n: 1 / n)
        dots(ax, rr, c, label=lab, marker=mk)
        curve(ax, (Ns[0], Ns[-1]), lambda n, k=k: th(n, k), c, lw=1.2, alpha=0.8)
    for y in (1 / PI_H, 2 / PI_H, 3 / PI_H):
        ax.axhline(y, color=MUTED, lw=0.8, ls=':', zorder=0)
    ax.set_ylim(0, 12)
    ax.legend(loc='upper right', fontsize=9, title='lines: exact Eq. 3 terms', title_fontsize=8.5)
    note(ax, 'dotted: $1/\\pi(\\frac{1}{2})$, $2/\\pi(\\frac{1}{2})$, $3/\\pi(\\frac{1}{2})$', loc=(0.98, 0.03))
    title(ax, 'E   the second step is the bottleneck: ⅔ of $\\langle M \\rangle$')

    ax = axC; loglog(ax, ylabel='mutations per run (mean, 95% CI)')
    for key, c, lab, mk in [('lost_neu', SKY, 'lost, neutral', 'o'),
                            ('fix_neu', ORANGE, 'fixed, neutral', 's'),
                            ('n_ben', PURP, 'beneficial, proposed', 'v'),
                            ('fix_ben', GREEN, 'beneficial, fixed', 'D'),
                            ('lost_del', VERM, 'lost, deleterious', '^')]:
        dots(ax, series('par_1e4', key), c, label=lab, marker=mk, ms=5.5)
    curve(ax, (Ns[0], Ns[-1]), lambda n: exact_M_parallel(n) / 1e4, ORANGE, ls='--', lw=1,
          label='exact $\\langle M \\rangle / N_P$')
    curve(ax, (Ns[0], Ns[-1]), lambda n: par_theory(n)['n_ben'], PURP, ls='--', lw=1,
          label='two-step theory, beneficial proposals')
    ax.axhline(2, color=GREEN, lw=1, ls='--', zorder=0)
    ax.legend(loc='upper left', fontsize=8.8, ncol=2)
    note(ax, 'no deleterious mutation fixes; fixed beneficial = 2 (the two steps)')
    title(ax, 'C   two beneficial fixations at every $N$; neutral fixations grow as $\\langle M\\rangle/N_P$')

    ax = axD; loglog(ax, ylabel='fraction of lost mutations (pooled over runs)')
    for k, c, lab, mk in [('lost_ben', GREEN, 'beneficial', 'D'), ('lost_del', VERM, 'deleterious', 's')]:
        dots(ax, pooled('par_1e4', k, 'lost'), c, label=lab, marker=mk)
    rn = pooled('par_1e4', 'lost_neu', 'lost')
    if len(rn):
        ax.text(0.02, 0.04, f'neutral: {rn[0, 1]:.3f} at $N = {int(rn[0, 0])}$, '
                f'{rn[-1, 1]:.6f} at $N = {int(rn[-1, 0]):,}$', transform=ax.transAxes, fontsize=9,
                color=SKY)
    rd = pooled('par_1e4', 'lost_del', 'lost')
    if len(rd):
        c0 = rd[len(rd) // 2, 1] * rd[len(rd) // 2, 0]
        curve(ax, (Ns[0], Ns[-1]), lambda n: c0 / n, MUTED, ls='--', lw=1, label='$\\propto 1/N$')
    ax.legend(loc='upper right', fontsize=9)
    title(ax, 'D   almost every lost mutation is neutral')
    save(fig, 'fig3_network_size.png', outdir)


# ── exact local neighbourhoods (Figs 4D, 5D) ──────────────────────────────────

def neighbourhood(genome):
    """Every single-mutation neighbour of a genotype under the engine's rule:
    list of (edge, computing?, [fitness of each acyclic neighbour])."""
    from faster_circuit.core import fitness, has_cycle_via
    from goals import and_funct
    g = list(genome)
    L = len(g); ng = (L - 1) // 2
    comp, stack = set(), [g[-1]]
    while stack:
        v = stack.pop()
        if v >= 2 and v not in comp:
            comp.add(v); stack += [g[2 * (v - 2)], g[2 * (v - 2) + 1]]
    out = []
    for e in range(L):
        old = g[e]
        if e == L - 1:
            srcs = [s for s in range(2, ng + 2) if s != old]
            is_c = True
        else:
            v = e // 2 + 2
            srcs = [s for s in range(ng + 2) if s not in (v, old) and not has_cycle_via(g, v, s)]
            is_c = v in comp
        fs = []
        for s in srcs:
            g[e] = s; fs.append(fitness(g, and_funct)); g[e] = old
        out.append((e, is_c, fs))
    return out, fitness(g, and_funct)


def draw_neighbourhood(ax, genome, label):
    """All neighbours of `genome`, one cluster per edge (computing edges: blue spokes)."""
    from matplotlib.patches import Patch
    nb, f0 = neighbourhood(genome)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.7, 1.55); ax.set_aspect('equal'); ax.axis('off')
    k = len(nb)
    n_c = n_cb = n_n = n_nb = 0
    for i, (e, is_c, fs) in enumerate(nb):
        th0 = np.pi / 2 + 2 * np.pi * i / k
        half = min(0.42, np.pi / k * 0.9)
        angs = th0 + (np.linspace(-half, half, len(fs)) if len(fs) > 1 else np.zeros(len(fs)))
        for a, f in zip(angs, fs):
            x, y = 1.1 * np.cos(a), 1.1 * np.sin(a)
            ax.plot([0, x], [0, y], color=SKY if is_c else '#c8c8c8', lw=0.8, zorder=1)
            ax.scatter([x], [y], s=34, c=FITC[f], edgecolors=GREEN if f > f0 else INK,
                       linewidths=1.6 if f > f0 else 0.5, zorder=3)
        if is_c:
            n_c += len(fs); n_cb += sum(f > f0 for f in fs)
        else:
            n_n += len(fs); n_nb += sum(f != f0 for f in fs)
    ax.scatter([0], [0], s=420, c=FITC[f0], edgecolors=INK, linewidths=1, zorder=4)
    ax.text(0, -1.42, f'{label}, fitness {f0:g}\n'
            f'via computing edges (blue): {n_c} neighbours, {n_cb} beneficial (green ring)\n'
            f'via non-computing edges: {n_n} neighbours, {n_nb} with a different fitness',
            ha='center', va='top', fontsize=8.8, color=MUTED)
    ax.legend(handles=[Patch(fc=c, ec=INK, lw=0.6, label=f'{f:g}') for f, c in FITC.items()],
              title='fitness', fontsize=8.5, title_fontsize=9, loc='upper left',
              bbox_to_anchor=(0.97, 1.0), frameon=True, edgecolor='#cccccc')


# ── figure 4: origin of linear scaling ────────────────────────────────────────

def draw_par_state(ax, edges, text):
    """Parallel-start network for Fig. 4A. edges: (src, dst, kind, new) with kind
    'b' (computing and beneficial to rewire, green), 'c' (computing, blue) or
    'n' (non-computing, black); new edges are dashed."""
    P = {'I1': (3.2, 8.7), 'I2': (6.4, 8.7), '1': (1.6, 5.4), '2': (4.2, 5.4), 'N': (8.6, 5.4),
         'O': (2.9, 2.2)}
    for k in ('I1', 'I2', 'O'):
        node(ax, P[k], {'I1': '$I_1$', 'I2': '$I_2$', 'O': '$O$'}[k], '#e3e3e3', w=1.3, h=0.9)
    for k in ('1', '2', 'N'):
        node(ax, P[k], k if k != 'N' else '$N$', w=1.1, h=0.9)
    ax.text(6.4, 5.4, '…', ha='center', va='center', fontsize=14)
    for a, b, kind, new in edges:
        c = {'b': GREEN, 'c': SKY, 'n': INK}[kind]
        lw, ls = (2.2 if kind != 'n' else 1.0), ('--' if new else '-')
        if P[a][1] == P[b][1]:                     # gate -> gate on the same row: side to side
            dup = sum(e[:2] == (a, b) for e in edges) > 1 and new
            ax.add_patch(FancyArrowPatch((P[a][0] - 0.55, P[a][1] + (0.25 if dup else 0)),
                                         (P[b][0] + 0.55, P[b][1] + (0.25 if dup else 0)),
                                         arrowstyle='-|>', mutation_scale=12, color=c, lw=lw, ls=ls,
                                         shrinkA=1, shrinkB=1, zorder=4,
                                         connectionstyle='arc3,rad=0.5' if dup else 'arc3'))
        else:
            arrow(ax, P[a], P[b], color=c, lw=lw, ls=ls, h=0.9)
    ax.text(4.9, 2.2, text, va='center', fontsize=9.5)


def fig4(outdir, big):
    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1, 1])
    ax = fig.add_subplot(gs[:, 0]); blank(ax)
    title(ax, 'A   the adaptive path from the parallel start')
    S = [([('I1', '1', 'b', 0), ('I2', '1', 'b', 0), ('I1', '2', 'n', 0), ('I2', '2', 'n', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('1', 'O', 'c', 0)],
          'fitness 0\n3 computing edges,\n2 beneficial to rewire'),
         ([('I1', '1', 'b', 0), ('2', '1', 'c', 1), ('I1', '2', 'c', 0), ('I2', '2', 'c', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('1', 'O', 'c', 0)],
          'fitness ½\n5 computing edges,\n1 beneficial to rewire'),
         ([('2', '1', 'c', 1), ('2', '1', 'c', 0), ('I1', '2', 'c', 0), ('I2', '2', 'c', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('1', 'O', 'c', 0)],
          'fitness 1: gate 1 =\nNAND(2, 2) = AND')]
    steps = ['$I_2 \\to 1$ rewired to a gate, e.g. $2 \\to 1$: $+\\frac{1}{2}$ (any of $N-1$ gates)',
             '$I_1 \\to 1$ rewired to a gate, e.g. $2 \\to 1$: $+\\frac{1}{2}$ (bottleneck)']
    h = 0.30
    for k, (edges, txt) in enumerate(S):
        sub = ax.inset_axes([0.0, 1 - (k + 1) * h - k * 0.035, 1.0, h]); blank(sub)
        draw_par_state(sub, edges, txt)
        if k < 2:
            yk = 1 - (k + 1) * h - k * 0.035 - 0.0175
            ax.annotate('', xy=(0.10, yk - 0.014), xytext=(0.10, yk + 0.014), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
            ax.text(0.15, yk, steps[k], transform=ax.transAxes, fontsize=9, va='center', color=MUTED)
    note(ax, 'green: computing edge with beneficial rewirings; blue: other computing edges;\n'
             'dashed: new edge. Neutral changes to non-computing edges happen in between\n'
             '(≈ $\\langle M \\rangle / N_P$ of them fix) and are not drawn.', loc=(0.5, -0.03),
         ha='center', va='top')

    axB, axC = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    axD, axE = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    Ns = grid('par_1e4'); xs = (Ns[0], Ns[-1])

    ax = axB; logx(ax, ylabel='computing edges')
    dots(ax, pooled('par_1e4', 'ce_sum', 'n_res'), BLUE, label='per resident network (pooled)')
    curve(ax, xs, lambda n: par_theory(n)['ce'], BLUE, lw=1.1, alpha=0.8)
    dots(ax, pooled('par_1e4', 'n_comp', 'M', scale=lambda n: 2 * n + 1), BLUE, hollow=True,
         marker='s', label='per proposed mutation: $(2N+1)\\,P(c)$')
    curve(ax, xs, lambda n: par_theory(n)['pc2'], BLUE, ls='--', lw=1.1, alpha=0.8)
    ax.set_ylim(0, 8)
    ax.legend(loc='upper left', fontsize=9, title='lines: two-step theory', title_fontsize=8.5)
    note(ax, 'per resident: 5 = (3 + 5 + 7)/3 while every resident is a step;\n'
             'neutral fixations (≈ $\\langle M\\rangle/N_P$) then weight residents by time: → 13/3')
    title(ax, 'B   computing edges stay $O(1)$')

    ax = axC; logx(ax, ylabel='fraction beneficial')
    dots(ax, pooled('par_1e4', 'fb_sum', 'fb_n', w='w_fb'), BLUE,
         label='structural $f_b$: beneficial rewirings per computing edge\n(landscape; pooled over residents, final excluded)')
    curve(ax, xs, lambda n: par_theory(n)['fb'], BLUE, lw=1.1, alpha=0.8)
    dots(ax, pooled('par_1e4', 'n_ben', 'n_comp'), GREEN, hollow=True, marker='s',
         label='$P(b\\,|\\,c)$: beneficial per proposal on a computing edge')
    curve(ax, xs, lambda n: par_theory(n)['pbc'], GREEN, ls='--', lw=1.1, alpha=0.8)
    ax.axhline(3 / 8, color=MUTED, lw=0.8, ls=':'); ax.axhline(4 / 13, color=MUTED, lw=0.8, ls=':')
    ax.text(1.0, 3 / 8, ' 3/8', transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED, va='center')
    ax.text(1.0, 4 / 13, ' 4/13', transform=ax.get_yaxis_transform(), fontsize=8.5, color=MUTED, va='center')
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper left', fontsize=8.8, title='lines: two-step theory', title_fontsize=8.5)
    note(ax, 'step 1: $f_b$ = (1, 1, 0); step 2: (1, 0, 0, 0, 0)\n'
             'per proposal the long second step dominates: 4/13')
    title(ax, 'C   beneficial fraction is $O(1)$')

    g = [0, 1] * 6 + [2]
    g[1] = 3                                           # gate 1 = NAND(I1, gate 2): fitness 1/2
    draw_neighbourhood(axD, g, 'state after step 1 ($N = 6$)')
    title(axD, 'D   all neighbours of the bottleneck state')

    ax = axE; loglog(ax, ylabel='probability per proposed mutation (pooled)')
    for (num_, den_, sc), k, c, lab, mk in [(('n_comp', 'M', None), 'pc', BLUE, '$P(c)$', 'o'),
                                            (('n_ben', 'n_comp', None), 'pbc', GREEN, '$P(b\\,|\\,c)$', 's'),
                                            (('fix_ben', 'n_ben', None), 'pfix', ORANGE, '$P_{fix}$ (beneficial)', 'D'),
                                            (('fix_ben', 'M', None), 'pb', PINK,
                                             '$P(b) = P(c)\\,P(b|c)\\,P_{fix}$', '^')]:
        dots(ax, pooled('par_1e4', num_, den_), c, label=lab, marker=mk)
        curve(ax, xs, lambda n, k=k: par_theory(n)[k], c, ls='--', lw=1.1, alpha=0.8)
    ax.legend(loc='lower left', fontsize=9, title='dashed: two-step theory', title_fontsize=8.5)
    note(ax, 'sums over runs;\n$\\langle M \\rangle = \\langle$fixed beneficial$\\rangle / P(b)$ exactly',
         loc=(0.98, 0.45), va='center')
    title(ax, 'E   Eq. 2 per proposal: only $P(c)$ depends on $N$')
    save(fig, 'fig4_mechanism.png', outdir)


# ── figure 5: chain start ─────────────────────────────────────────────────────

def fig5(outdir, big):
    from bottleneck_replication.starts import chain
    fig = plt.figure(figsize=(23, 10.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.75, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0]); blank(ax)
    draw_chain(ax)
    title(ax, 'A   chain start (every edge computing)')
    note(ax, 'gate 1 = NAND$(I_1, I_2)$, gates 2…N = NOT of the\nprevious; odd $N$: computes NAND,'
             ' fitness 0\ngate $k$ can take only $k$ sources without a cycle\n'
             f'$N_P = 10^4$; {n_runs("chain_1e4")} runs per $N$', loc=(0.02, 0.0), ha='left')
    Ns = grid('chain_1e4'); xs = (Ns[0], Ns[-1])

    ax = fig.add_subplot(gs[0, 1]); loglog(ax, ylabel='computing edges')
    curve(ax, xs, lambda n: 2 * n + 1, '#bdbdbd', lw=5, label='$2N+1$ (all edges)')
    dots(ax, pooled('chain_1e4', 'ce_sum', 'n_res'), VERM, label='per resident network (pooled)')
    dots(ax, pooled('chain_1e4', 'n_comp', 'M', scale=lambda n: 2 * n + 1), VERM, hollow=True, marker='s',
         label='per proposed mutation: $(2N+1)\\,P(c)$', offset=1.12)
    ax.legend(loc='upper left', fontsize=9)
    title(ax, 'B   the module stays (almost) the whole network')

    ax = fig.add_subplot(gs[0, 2]); logx(ax, ylabel='fraction beneficial')
    dots(ax, pooled('chain_1e4', 'fb_sum', 'fb_n', w='w_fb'), VERM,
         label='structural $f_b$ (landscape; pooled over residents)')
    dots(ax, pooled('chain_1e4', 'n_ben', 'n_comp'), GREEN, hollow=True, marker='s',
         label='$P(b\\,|\\,c)$ per proposal (pooled)', offset=1.08)
    dots(ax, series('chain_1e4', 'pbc_mut', stat='median'), GREEN, hollow=True, marker='D', ms=5,
         label='$P(b\\,|\\,c)$, typical run (median)', offset=0.92)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=8.8)
    note(ax, 'per proposal is lower: runs spend most proposals\nat fitness ¾, where few rewirings help',
         loc=(0.02, 0.03), ha='left')
    title(ax, 'C   beneficial fraction is $O(1)$')

    ax = fig.add_subplot(gs[0, 3])
    draw_neighbourhood(ax, chain(7), 'chain start ($N = 7$)')
    title(ax, 'D   all neighbours of the chain start')

    ax = fig.add_subplot(gs[1, 1]); loglog(ax, ylabel='total mutations proposed, $M$')
    rm, r = series('chain_1e4', 'M', stat='median'), series('chain_1e4', 'M')
    dots(ax, rm, VERM, label='median (95% CI; thin: 5–95% of runs)', spread=True)
    dots(ax, r, VERM, label='mean (95% CI)', hollow=True, marker='s', offset=1.12)
    ax.plot(Ns, [inv_p0_formula(n, n) for n in Ns], color=MUTED, ls='--', lw=1.1,
            label='exact first step $1/p_0$ (Eq. 4, $L = N$)')
    ax.legend(loc='upper left', fontsize=9)
    sh = [(N, top_share('chain_1e4', N)) for N in Ns]
    Nx, sx = max(sh, key=lambda t: t[1])
    note(ax, f'longest 1% of runs: up to {sx:.0%} of all proposals (at $N = {Nx:,}$)')
    title(ax, 'E   typical number of mutations does not grow with $N$')

    ax = fig.add_subplot(gs[1, 2]); loglog(ax, ylabel='probability per proposed mutation (pooled)')
    for num_, den_, c, lab, mk in [('n_comp', 'M', BLUE, '$P(c)$', 'o'),
                                   ('n_ben', 'n_comp', GREEN, '$P(b\\,|\\,c)$', 's'),
                                   ('fix_ben', 'n_ben', ORANGE, '$P_{fix}$ (beneficial)', 'D'),
                                   ('fix_ben', 'M', PINK, '$P(b)$', '^')]:
        dots(ax, pooled('chain_1e4', num_, den_), c, label=lab, marker=mk)
    ax.set_ylim(0.01, 2); ax.keep_ylim = True
    ax.legend(loc='lower left', fontsize=9, ncol=2)
    rb, rf = series('chain_1e4', 'n_ben'), series('chain_1e4', 'fix_ben')
    if len(rb):
        note(ax, f'beneficial proposals per run: {rb[:, 1].min():.1f}–{rb[:, 1].max():.1f}; '
                 f'fixed: {rf[:, 1].min():.2f}–{rf[:, 1].max():.2f}', loc=(0.98, 0.97), va='top')
    title(ax, 'F   Eq. 2 per proposal: no factor depends on $N$')

    ax = fig.add_subplot(gs[1, 3]); loglog(ax, ylabel='fraction of runs with $M > x$', xlabel='$x$')
    cm = plt.get_cmap('viridis')
    show = [N for N in Ns if N in (5, 101, 1001, 10001, 1000001)] or Ns[::4]
    for i, N in enumerate(show):
        M = np.sort(load('chain_1e4', N)['M'])
        ax.step(M, 1 - np.arange(len(M)) / len(M), where='post', color=cm(i / max(len(show) - 1, 1)),
                lw=1.4, label=f'chain, $N = {N:,}$')
    for N in [n for n in grid('par_1e4') if n in (100, 10000)]:
        M = np.sort(load('par_1e4', N)['M'])
        ax.step(M, 1 - np.arange(len(M)) / len(M), where='post', color=GREY, lw=1, ls='--',
                label=f'parallel, $N = {N:,}$')
    xx = np.logspace(2, 5, 20)
    ax.plot(xx, 5 / xx, color=INK, ls=':', lw=1.2, label='$\\propto 1/x$')
    ax.set_ylim(5e-4, 1.2); ax.keep_ylim = True
    ax.legend(loc='lower left', fontsize=8.5)
    title(ax, 'G   one distribution at every $N$, with a heavy tail')
    save(fig, 'fig5_chain_start.png', outdir)


# ── supplementary figures ─────────────────────────────────────────────────────

def figS1(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax = axs[0]; logx(ax, ylabel='mutations proposed / $N$ (mean, 95% CI)')
    r = series('rw', 'Mn')
    nmax = r[-1, 0] if len(r) else None
    dots(ax, series('par_1e4', 'Mn', nmax=nmax), BLUE, hollow=True, marker='s',
         label='with selection, $N_P = 10^4$ (Fig. 3)')
    curve(ax, (5, nmax or 1e5), lambda n: exact_M_parallel(n) / n, BLUE, lw=1.1, alpha=0.8,
          label='exact theory with selection (Eq. 3)')
    dots(ax, r, PURP, label='random walk (every mutation accepted)')
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=9)
    note(ax, '$M$ counts proposals, not time. The exact theory assumes selection;\n'
             'the random walk has none (it also steps down in fitness).', loc=(0.02, 0.03), ha='left')
    title(ax, 'A   random walk from the parallel start: $M \\propto N$')

    ax = axs[1]; loglog(ax, ylabel='probability per proposed mutation (pooled)')
    for exp, hollow, tag in [('rw', False, 'random walk'), ('par_1e4', True, 'selection')]:
        dots(ax, pooled(exp, 'n_comp', 'M', nmax=nmax), BLUE, hollow=hollow, marker='o',
             label=f'$P(c)$, {tag}')
        dots(ax, pooled(exp, 'n_ben', 'n_comp', nmax=nmax), GREEN, hollow=hollow, marker='s',
             label=f'$P(b\\,|\\,c)$, {tag}')
    rp = pooled('rw', 'ce_sum', 'n_res', scale=lambda n: 1 / (2 * n + 1))
    dots(ax, rp, GREY, marker='.', ms=7, ci=False, label='random walk: computing edges / $(2N+1)$')
    ax.legend(loc='lower left', fontsize=8.5, ncol=2)
    note(ax, 'without selection the module is a fixed fraction of the network\n'
             'and the beneficial fraction falls instead. $P(c)$ sits below the edge\n'
             'fraction: rewiring a computing gate closes a cycle more often (redrawn)',
         loc=(0.02, 0.19), ha='left')
    ax.set_ylim(1e-7, 3); ax.keep_ylim = True
    title(ax, 'B   the factors of Eq. 2 swap roles')
    save(fig, 'figS1_random_walk.png', outdir)


def figS2(outdir, big):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, N, lab in [(axs[0, 0], 40, 'A'), (axs[0, 1], 400, 'B')]:
        d = load('par_1e4', N)
        M = d['M']
        top = np.percentile(M, 99.5)
        bins = np.linspace(0, top, 30)
        ax.hist(M[M <= top], bins=bins, color=BLUE, alpha=0.85, ec='white', lw=0.5)
        t1, t2 = exact_M_parallel_terms(N)
        pmf = geom_sum_pmf(t1, t2, int(bins[-1]) + 2)
        cum = np.concatenate([[0], np.cumsum(pmf)])
        exp_counts = len(M) * (np.interp(bins[1:], np.arange(len(cum)), cum)
                               - np.interp(bins[:-1], np.arange(len(cum)), cum))
        ax.step(bins, np.r_[exp_counts, exp_counts[-1]], where='post', color=PINK, lw=1.8,
                label='exact: two geometric waits (Eq. 3 terms)')
        ax.axvline(M.mean(), color=INK, lw=1.4, label=f'mean {M.mean():.0f}')
        ax.axvline(np.median(M), color=INK, lw=1.4, ls='--', label=f'median {np.median(M):.0f}')
        ax.set_xlim(0, top)
        ax.set_xlabel('mutations proposed, $M$'); ax.set_ylabel('number of runs')
        ax.legend(fontsize=9.5)
        note(ax, f'{(M > top).sum()} runs beyond the axis (max {M.max():,.0f})', loc=(0.98, 0.55), va='top')
        title(ax, f'{lab}   distribution at $N = {N}$ ({len(M)} runs)')
    ax = axs[1, 0]
    ax.set_yscale('log'); ax.set_xlabel('$x = M / \\langle M \\rangle$')
    ax.set_ylabel('fraction of runs with $M / \\langle M\\rangle > x$'); ax.grid(True, color=GRID_C, lw=0.6)
    Ns = grid('par_1e4')
    cm = plt.get_cmap('viridis')
    for i, N in enumerate(Ns):
        M = np.sort(load('par_1e4', N)['M'])
        ax.step(M / M.mean(), 1 - np.arange(len(M)) / len(M), where='post', color=cm(i / max(len(Ns) - 1, 1)),
                lw=1.1, label=f'$N={N:,}$' if i in (0, len(Ns) // 2, len(Ns) - 1) else None)
    x = np.linspace(0, 6, 200)
    ax.plot(x, 2 * np.exp(-1.5 * x) - np.exp(-3 * x), color=PINK, lw=2,
            label='exact, large $N$: $2e^{-3x/2} - e^{-3x}$')
    ax.plot(x, np.exp(-x), color=INK, ls=':', lw=1, alpha=0.6, label='single exponential $e^{-x}$')
    ax.set_ylim(1e-3, 1.3); ax.set_xlim(0, 6); ax.keep_ylim = True
    ax.legend(fontsize=9)
    note(ax, 'right-skewed but lighter-tailed than exponential\n(coefficient of variation → √5/3 ≈ 0.75)',
         loc=(0.98, 0.70), va='top')
    title(ax, 'C   one shape at every $N$: two exponential waits')

    ax = axs[1, 1]; logx(ax, ylabel='mutations proposed / $N$')
    rm = series('par_1e4', 'Mn'); rmed = series('par_1e4', 'Mn', stat='median')
    dots(ax, rm, BLUE, label='mean (95% CI)')
    dots(ax, rmed, BLUE, hollow=True, marker='s', label='median (95% CI)', offset=1.1)
    curve(ax, (Ns[0], Ns[-1]), lambda n: exact_M_parallel(n) / n, BLUE, lw=1.1, label='exact mean (Eq. 3)')
    curve(ax, (Ns[0], Ns[-1]),
          lambda n: np.array([hypo_median(*exact_M_parallel_terms(k)) for k in n]) / n,
          BLUE, ls='--', lw=1.1, label='exact median (two exponential waits)')
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=9)
    note(ax, 'paper: median $7.8\\,N^{0.96}$ ($N \\leq 10^3$); median/mean → 0.82 at every large $N$',
         loc=(0.02, 0.03), ha='left')
    title(ax, 'D   median and mean scale alike')
    save(fig, 'figS2_distributions.png', outdir)


def _start_fitness(target):
    return bin(~(0b0111 ^ target) & 0xF).count('1') / 4


def figS3(outdir, big):
    from bottleneck_replication.starts import parallel
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
    for k, (ax, (exp, lab, tgt)) in enumerate(zip(axs, [
            ('andnot_10', '$I_1$ AND NOT $I_2$', ANDNOT), ('or_10', 'OR', OR), ('true_10', 'TRUE', TRUE)])):
        logx(ax, ylabel='mutations proposed / $N$ (mean, 95% CI)')
        r = series(exp, 'Mn')
        dots(ax, r, PINK, label='simulation, $N_P = 10$')
        ax.set_ylim(0, None)
        p0, _ = enumerate_p(parallel(20), 10.0, tgt)
        if p0 > 0:
            d100 = load(exp, 100)
            ip = 1 / enumerate_p(parallel(100), 10.0, tgt)[0]
            txt = (f'from the start, one rewiring in $N(2N+1)$ is beneficial:\n'
                   f'with the start frozen, $1/p_0 \\propto N^2$ ({ip:,.0f} at $N = 100$'
                   + (f', vs $\\langle M\\rangle$ = {d100["M"].mean():,.0f})' if d100 else ')'))
        else:
            txt = 'no single mutation of the start is beneficial ($p_0 = 0$):\nthe first step must be neutral or deleterious'
        note(ax, txt + '\n→ the frozen-genotype theory does not describe these runs', loc=(0.02, 0.03), ha='left')
        ax.legend(loc='upper right', fontsize=9)
        title(ax, f'{"ABC"[k]}   NAND → {lab} (start fitness {_start_fitness(tgt):g})')
    save(fig, 'figS3_other_targets.png', outdir)


def small_pop(outdir, big, exp, exp_big, color, name, titles, chain_start=False):
    """Figs S4 / S5: N_P = 10, compared with N_P = 10^4."""
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 10.5))
    Ns = grid(exp); xs = (Ns[0], Ns[-1])
    ax = axs[0, 0]; loglog(ax, ylabel='total mutations proposed, $M$')
    if chain_start:
        dots(ax, series(exp, 'M', stat='median'), color, label='median (95% CI; thin: 5–95% of runs)',
             spread=True)
        dots(ax, series(exp, 'M'), color, label='mean (95% CI)', hollow=True, marker='s', offset=1.12)
        ax.plot(Ns, [inv_p0_formula(n, n, 10.0) for n in Ns], color=MUTED, ls='--', lw=1.1,
                label='exact first step $1/p_0$ (Eq. 4, $L = N$)')
        sh = [(N, top_share(exp, N)) for N in Ns]
        Nx, sx = max(sh, key=lambda t: t[1])
        note(ax, f'longest 1% of runs: up to {sx:.0%} of all proposals (at $N = {Nx:,}$)')
    else:
        r = series(exp, 'M')
        dots(ax, r, color, label='mean (95% CI; thin: 5–95% of runs)', spread=True)
        curve(ax, xs, lambda n: exact_M_parallel(n, 10.0), MUTED, ls='--', lw=1.1,
              label='exact Eq. 3 at $N_P = 10$ (two-step path, no drift)')
        ratio = r[:, 1] / exact_M_parallel(r[:, 0], 10.0)
        note(ax, f'simulation / Eq. 3 at $N_P = 10$: {ratio.min():.2f}–{ratio.max():.2f}, flat in $N$\n'
                 '(drift scrambles the reservoir and adds detours)')
    ax.legend(loc='upper left', fontsize=9)
    title(ax, titles[0])

    ax = axs[0, 1]; loglog(ax, ylabel='fixed mutations per run (mean, 95% CI)')
    for key, c, lab, mk in [('fix_neu', SKY, 'neutral', 'o'), ('fix_ben', GREEN, 'beneficial', 'D'),
                            ('fix_del', VERM, 'deleterious', 's')]:
        dots(ax, series(exp, key), c, label=f'$N_P = 10$, {lab}', marker=mk, ms=5.5)
    dots(ax, series(exp_big, 'fix_ben', nmax=xs[1]), GREEN, hollow=True, marker='D', ms=5,
         label='$N_P = 10^4$, beneficial', offset=1.12)
    rM = series(exp, 'M')
    ax.plot(rM[:, 0], rM[:, 1] / 10, color=SKY, ls='--', lw=1, label='$\\langle M\\rangle / N_P$')
    ax.legend(loc='upper left', fontsize=8.8)
    title(ax, titles[1])

    ax = axs[1, 0]
    if chain_start:
        logx(ax, ylabel='$P(c)$ per proposed mutation'); ax.set_ylim(0, 1.05)
    else:
        loglog(ax, ylabel='$P(c)$ per proposed mutation')
    dots(ax, pooled(exp, 'n_comp', 'M'), color, label='$N_P = 10$, pooled')
    dots(ax, series(exp, 'pc_mut', stat='median'), color, hollow=True, marker='D', ms=5,
         label='$N_P = 10$, typical run (median)', offset=1.1)
    dots(ax, pooled(exp_big, 'n_comp', 'M', nmax=xs[1]), INK, hollow=True, marker='s', ms=5,
         label='$N_P = 10^4$, pooled', offset=0.9)
    ax.legend(loc='lower left' if not chain_start else 'lower right', fontsize=9)
    title(ax, titles[2])

    ax = axs[1, 1]; logx(ax, ylabel='fraction beneficial')
    dots(ax, pooled(exp, 'n_ben', 'n_comp'), color, label='$P(b\\,|\\,c)$, $N_P = 10$, pooled')
    dots(ax, series(exp, 'pbc_mut', stat='median'), color, hollow=True, marker='D', ms=5,
         label='$P(b\\,|\\,c)$, $N_P = 10$, typical run (median)', offset=1.1)
    dots(ax, pooled(exp_big, 'n_ben', 'n_comp', nmax=xs[1]), INK, hollow=True, marker='s', ms=5,
         label='$P(b\\,|\\,c)$, $N_P = 10^4$, pooled', offset=0.9)
    dots(ax, pooled(exp, 'fb_sum', 'fb_n', w='w_fb'), GREY, marker='.', ms=8, ci=False,
         label='structural $f_b$, $N_P = 10$ (landscape)')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8.5)
    title(ax, titles[3])
    save(fig, name, outdir)


def figS4(outdir, big):
    small_pop(outdir, big, 'par_10', 'par_1e4', ORANGE, 'figS4_small_population.png',
              ['A   parallel start, $N_P = 10$: $M \\propto N$', 'B   fixations: drift adds $\\langle M\\rangle/N_P$',
               'C   $P(c)$', 'D   $P(b\\,|\\,c)$'])


def figS5(outdir, big):
    small_pop(outdir, big, 'chain_10', 'chain_1e4', GREEN, 'figS5_small_population_chain.png',
              ['A   chain start, $N_P = 10$: typical $M$ flat', 'B   fixations: drift adds $\\langle M\\rangle/N_P$',
               'C   $P(c) \\approx 1$ at $N_P = 10^4$; below 1 and noisy under drift', 'D   $P(b\\,|\\,c)$'], chain_start=True)


def figS6(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.8))
    ax = axs[0]; logx(ax, ylabel='time to evolve AND, $T\\,N_P/\\tau$')
    r = series('par_1e4', 'Tn')
    Ns = grid('par_1e4')
    dots(ax, r, BLUE, label='mean (95% CI)')
    curve(ax, (Ns[0], Ns[-1]), lambda n: exact_M_parallel(n) / (2 * n + 1), BLUE, lw=1.1,
          label='exact: Eq. 3 / $(2N+1)$')
    ax.axhline(3 / (2 * PI_H), color=MUTED, ls=':', lw=1, label=f'limit $3/(2\\pi(\\frac{{1}}{{2}}))$ = {3 / (2 * PI_H):.2f}')
    ax.set_ylim(0, 6)
    ax.legend(loc='upper right', fontsize=9)
    title(ax, 'A   parallel start: time is flat in $N$')
    ax = axs[1]; loglog(ax, ylabel='time to evolve AND, $T\\,N_P/\\tau$')
    rm = series('chain_1e4', 'Tn', stat='median')
    dots(ax, rm, VERM, label='median (95% CI; thin: 5–95% of runs)', spread=True)
    dots(ax, series('chain_1e4', 'Tn'), VERM, label='mean (95% CI)', hollow=True, marker='s', offset=1.12)
    if len(rm):
        i = len(rm) // 2
        c0 = rm[i, 1] * rm[i, 0]
        curve(ax, (rm[0, 0], rm[-1, 0]), lambda n: c0 / n, MUTED, ls='--', lw=1, label='$\\propto 1/N$')
    ax.legend(loc='lower left', fontsize=9)
    title(ax, 'B   chain start: time falls as $1/N$')
    note(axs[0], 'time model (Methods 6.1–6.2): one exponential wait per accepted (acyclic)\n'
                 'proposal, rate $N_P(2N+1)/\\tau$; cyclic redraws take no time; fixation\n'
                 '(sweep) time ignored. Shown: $E[T\\,|\\,M] = \\tau M / (N_P(2N+1))$ per run.',
         loc=(0.02, 0.03), ha='left')
    save(fig, 'figS6_time.png', outdir)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global NMAX
    which = sys.argv[1] if len(sys.argv) > 1 else '1e4'
    NMAX = int(float(which)) + 1
    big = NMAX > 2 * 10**4
    outdir = os.path.join(HERE, f'figs_{which}')
    os.makedirs(outdir, exist_ok=True)
    only = sys.argv[2:]
    for name, fn, needs_big in [('fig1', fig1, False), ('fig2', fig2, False), ('fig3', fig3, True),
                                ('fig4', fig4, True), ('fig5', fig5, True), ('figS1', figS1, True),
                                ('figS2', figS2, True), ('figS3', figS3, True), ('figS4', figS4, True),
                                ('figS5', figS5, True), ('figS6', figS6, True)]:
        if only and name not in only:
            continue
        try:
            fn(outdir, big) if needs_big else fn(outdir)
        except Exception as e:                     # missing data for a figure: report, go on
            import traceback
            traceback.print_exc()
            print(f'skipped {name}: {type(e).__name__}: {e}')
            plt.close('all')


if __name__ == '__main__':
    main()
