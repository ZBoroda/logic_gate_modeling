"""
Every figure of the paper (arXiv 2610.40635), redrawn from faster_circuit runs.

  python -m paper_replication.figures 1e4    ->  paper_replication/figs_1e4/
  python -m paper_replication.figures 1e6    ->  paper_replication/figs_1e6/

The 1e4 set shows N <= 10^4 (the paper's range, extended tenfold) with fits over
the plotted points. The 1e6 set shows everything simulated (N_P = 10^4 up to
10^6; N_P = 10 and the random walk as far as they were run) and adds a fit over
N >= 10^4. As in the paper: dots are means over runs, bars are 5th-95th
percentiles of the per-run values, fits are y = A N^b by least squares on
log(mean) vs log N.
"""
import os
import sys
import math
import itertools
import zlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_replication.run import EXPS, path, AND, ANDNOT, OR, TRUE, FB_DEF
from bottleneck_replication.theory import exact_M_parallel

HERE = os.path.dirname(__file__)
INK, MUTED, GRID_C = '#1a1a1a', '#6b6b6b', '#e6e6e6'
BLUE, ORANGE, GREEN, PINK, SKY, VERM, YEL, PURP = ('#0072B2', '#E69F00', '#009E73', '#CC79A7',
                                                  '#56B4E9', '#D55E00', '#F0E442', '#2E2A7A')
# fitness categories, viridis as in the paper's Figs 2, 4D, 5D
FITC = {0.0: '#440154', 0.25: '#3b528b', 0.5: '#21918c', 0.75: '#5ec962', 1.0: '#fde725'}
TAU = 1000.0                     # 1 / mutation rate per edge per network (Methods 6.1)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#333', 'axes.linewidth': 0.9, 'legend.frameon': False,
                     'savefig.dpi': 150, 'axes.titlesize': 12.5})

NMAX = 10**4
FIT_HI_LO = 10**4


# ── data ──────────────────────────────────────────────────────────────────────

def load(exp, N):
    p = path(exp, N)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    st = d['stats']
    with np.errstate(invalid='ignore', divide='ignore'):
        out = dict(M=st[:, 0], fix=st[:, 1:4].sum(1), lost=st[:, 4:7].sum(1),
                   fix_ben=st[:, 1], lost_ben=st[:, 4], lost_del=st[:, 5], lost_neu=st[:, 6],
                   ce=st[:, 8] / st[:, 7], fb=st[:, 9] / st[:, 10], capped=st[:, 11],
                   counts=d['counts'])
    # mutation-weighted counts (files from before the n_comp column: nan)
    out['n_comp'] = st[:, 12] if st.shape[1] > 12 else np.full(len(st), np.nan)
    out['n_ben'] = st[:, 1] + st[:, 4]
    with np.errstate(invalid='ignore', divide='ignore'):
        out['pc_mut'] = out['n_comp'] / out['M']
        out['pbc_mut'] = out['n_ben'] / out['n_comp']
    if str(d.get('fb_def', '')) != FB_DEF:            # f_b from an older convention:
        out['fb'] = np.full(len(st), np.nan)        # never plot it
    out['pc'] = out['ce'] / (2 * N + 1)
    out['frac_ben'] = out['lost_ben'] / out['lost']
    out['frac_del'] = out['lost_del'] / out['lost']
    out['frac_neu'] = out['lost_neu'] / out['lost']
    Np = EXPS[exp][2] or 1.0
    rng = np.random.default_rng([zlib.crc32(exp.encode()), N])
    out['T'] = rng.gamma(np.maximum(out['M'], 1e-9), TAU / ((2 * N + 1) * Np))
    return out


def series(exp, key, nmax=None, stat='mean'):
    """Rows (N, centre, p5, p95, sem, n) over the experiment's N grid."""
    nmax = nmax or NMAX
    rows = []
    for N in EXPS[exp][3]:
        if N > nmax * 1.0001 + 1:
            continue
        d = load(exp, N)
        if d is None:
            continue
        x = d[key]
        x = x[np.isfinite(x)]
        if not len(x):
            continue
        c = np.median(x) if stat == 'median' else x.mean()
        rows.append((N, c, np.percentile(x, 5), np.percentile(x, 95),
                     x.std(ddof=1) / math.sqrt(len(x)) if len(x) > 1 else 0, len(x)))
    return np.array(rows) if rows else np.zeros((0, 6))


def fit(r, lo=0, hi=np.inf, nmin=3):
    """y = A N^b on rows with lo <= N <= hi. Returns (A, b, b_se)."""
    m = (r[:, 0] >= lo) & (r[:, 0] <= hi) & (r[:, 1] > 0)
    if m.sum() < nmin:
        return None
    x, y = np.log(r[m, 0]), np.log(r[m, 1])
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    se = math.sqrt((res ** 2).sum() / max(len(x) - 2, 1) / ((x - x.mean()) ** 2).sum())
    return math.exp(a), b, se


def halves(r):
    """Exponents fitted on the lower and upper halves of the points (paper style)."""
    n = len(r)
    if n < 6:
        return None
    k = n // 2
    lo, hi = fit(r[:k]), fit(r[k:])
    return lo[1], hi[1]


def thirds(r, hi):
    """Paper's sensitivity check: fit 3 non-overlapping contiguous subsets of the
    points with N <= hi. Returns (A mean, A sd, b mean, b sd)."""
    rr = r[r[:, 0] <= hi * 1.001]
    fs = [fit(c, nmin=2) for c in np.array_split(rr, 3)]
    fs = [f for f in fs if f]
    if len(fs) < 3:
        return None
    A = np.array([f[0] for f in fs]); b = np.array([f[1] for f in fs])
    return A.mean(), A.std(ddof=1), b.mean(), b.std(ddof=1)


def b_on(r, lo, hi):
    f = fit(r, lo, hi * 1.001, nmin=2)
    return f[1] if f else float('nan')


def num(x, fmt='{:.2f}'):
    """Number with a typographic minus sign."""
    return fmt.format(x).replace('-', '−')


def esc(t):
    return t.replace('{', '{{').replace('}', '}}')


def tex_n(n):
    k = math.log10(n)
    return f'10^{{{int(round(k))}}}' if abs(k - round(k)) < 1e-9 else f'{n:g}'


# ── drawing helpers ───────────────────────────────────────────────────────────

def title(ax, s):
    ax.set_title(s, loc='left', fontweight='bold')


def loglog(ax, xlabel='network size $N$', ylabel=None):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which='major', color=GRID_C, lw=0.6, zorder=0)


def dots(ax, r, color, label=None, marker='o', hollow=False, offset=1.0, ms=6.5, bars=True):
    if not len(r):
        return
    x = r[:, 0] * offset
    if bars:
        lo = np.clip(r[:, 1] - r[:, 2], 0, None)
        hi = np.clip(r[:, 3] - r[:, 1], 0, None)
        ax.errorbar(x, r[:, 1], yerr=[lo, hi], fmt='none', ecolor=color, elinewidth=1.1,
                    capsize=0, alpha=0.75, zorder=2)
    kw = dict(mfc='white', mec=color, mew=1.6) if hollow else dict(mfc=color, mec=INK, mew=0.6)
    ax.plot(x, r[:, 1], marker, ms=ms, ls='none', label=label, zorder=3, **kw)


def fitline(ax, r, color, lo=0, hi=np.inf, ls='-', label_fmt='fit $y = {A:.3g}\\,N^{{{b:.2f}}}$',
            span=None, lw=1.4):
    f = fit(r, lo, hi)
    if f is None:
        return None
    A, b, _ = f
    m = (r[:, 0] >= lo) & (r[:, 0] <= hi)
    xs = np.array(span or (r[m, 0].min(), r[m, 0].max()))
    xx = np.logspace(np.log10(xs[0]), np.log10(xs[1]), 50)
    ax.plot(xx, A * xx ** b, color=color, lw=lw, ls=ls, zorder=1,
            label=label_fmt.format(A=A, b=b) if label_fmt else None)
    return f


def fit_lines(ax, r, color, big):
    """Fit over all points; in the 1e6 set, also over N >= 10^4."""
    f = fitline(ax, r, color, label_fmt='all $N$: $y={A:.3g}\\,N^{{{b:.2f}}}$')
    if big and (r[:, 0] >= FIT_HI_LO).sum() >= 3:
        fitline(ax, r, color, lo=FIT_HI_LO, ls='--',
                label_fmt='$N \\geq 10^4$: $y={A:.3g}\\,N^{{{b:.2f}}}$')
    return f


def paper_fits(ax, r, color, big, paper_hi, lw=1.4):
    """Solid fit over the paper's N range (comparable with its numbers), dashed fit
    over the extension: N >= paper_hi in the 1e4 set, N >= 10^4 in the 1e6 set."""
    f = fitline(ax, r, color, hi=paper_hi * 1.001, lw=lw,
                label_fmt=f'$N \\leq {esc(tex_n(paper_hi))}$ (paper range): ' + '$y={A:.3g}\\,N^{{{b:.2f}}}$')
    lo = FIT_HI_LO if big and (r[:, 0] >= FIT_HI_LO).sum() >= 3 else paper_hi
    g = None
    if (r[:, 0] >= lo).sum() >= 3:
        g = fitline(ax, r, color, lo=lo, ls='--', lw=lw,
                    label_fmt=f'$N \\geq {esc(tex_n(lo))}$: ' + '$y={A:.3g}\\,N^{{{b:.2f}}}$')
    return f, g


def note(ax, s, loc=(0.98, 0.04), ha='right', va='bottom', fs=9):
    ax.text(*loc, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=MUTED)


def _limits(ax):
    """Log-axis limits from all plotted data (lines and error bars), with margin."""
    from matplotlib.collections import LineCollection
    xs, ys = [], []
    for ln in ax.get_lines():
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if ln.get_label() == '_ref':
            continue
        xs.append(x); ys.append(y)
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
    ax.text(8.4, 6.0, 'select edge to mutate', ha='center', fontsize=10)
    ax.text(8.4, 4.9, '$I_1 \\to 1$ (red) is removed', ha='center', fontsize=9.5, color=VERM)
    ax.annotate('', xy=(21.9, 5.6), xytext=(17.9, 5.6), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.3))
    ax.text(19.9, 6.0, 'create new edge', ha='center', fontsize=10)
    ax.text(19.9, 4.9, 'choices for new edge:\n$I_2 \\to 1$,  $2 \\to 1$ (green)', ha='center',
            va='top', fontsize=9.5)
    ax.text(0.2, 0.0, 'new source: any node except the old one, the gate itself and $O$ '
                      '(for the edge into $O$: any other gate); proposals that close a cycle are redrawn',
            fontsize=9, color=MUTED)

    # C: flowchart (paper Fig. 1C)
    ax = fig.add_subplot(gs[0, 1]); blank(ax)
    title(ax, 'C   evolutionary algorithm (SSWM, Moran fixation)')
    box(ax, 0.2, 8.4, 4.4, 9.5, 'initialize resident network\nof size $N$')
    box(ax, 0.2, 6.0, 4.4, 7.3, 'mutate the genotype &\ncompute fitness $f$')
    box(ax, 0.2, 3.2, 4.4, 4.6, 'does the mutation fixate?\n'
                                '$\\pi(s) = \\frac{1-e^{-s}}{1-e^{-N_P s}}$', fs=9.2)
    box(ax, 5.4, 4.8, 9.0, 5.8, 'keep resident\ngenotype as it is')
    box(ax, 5.4, 2.0, 9.0, 3.0, 'update resident to\nthe mutated genotype')
    box(ax, 6.2, 0.2, 8.2, 0.95, 'end', fc='#fdf3e3')
    poly_arrow(ax, [(2.3, 8.4), (2.3, 7.3)])
    poly_arrow(ax, [(2.3, 6.0), (2.3, 4.6)])
    poly_arrow(ax, [(4.4, 3.9), (4.9, 3.9), (4.9, 5.3), (5.4, 5.3)], label='no', lpos=(4.65, 4.95))
    poly_arrow(ax, [(4.4, 3.9), (4.9, 3.9), (4.9, 2.5), (5.4, 2.5)], label='yes', lpos=(4.65, 2.6))
    poly_arrow(ax, [(9.0, 5.3), (9.35, 5.3), (9.35, 6.45), (4.4, 6.45)])
    poly_arrow(ax, [(9.0, 2.5), (9.75, 2.5), (9.75, 6.85), (4.4, 6.85)])
    ax.text(9.9, 4.0, 'if $f \\neq 1$', fontsize=9, rotation=90, ha='left', va='center')
    poly_arrow(ax, [(7.2, 2.0), (7.2, 0.95)], label='', lpos=None)
    ax.text(7.35, 1.3, 'if $f = 1$', fontsize=9, ha='left', va='center')
    ax.text(0.2, -0.45, '$M$ counts every accepted (acyclic) mutation, fixed or lost',
            fontsize=9, color=MUTED)
    save(fig, 'fig1_model.png', outdir)


# ── figure 2: genotype graph of size-2 networks ───────────────────────────────

def fig2(outdir):
    """Genotype graph from the repo's gen_fig2.py (unordered input pairs, as in the
    paper: 54 genotypes), restyled."""
    import networkx as nx
    from matplotlib.patches import Patch
    import gen_fig2 as g2
    from fast_circuit.circuit import _target_array
    from goals import and_funct
    g2.warmup()
    target = _target_array(and_funct)
    genotypes, p0, p1, outs = g2.enumerate_genotypes()
    fit_ = {g: g2.compute_fitness(*g, target) for g in genotypes}
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
    title(ax, f'Genotype graph of all acyclic networks with $N = 2$ '
              f'({G.number_of_nodes()} genotypes, {G.number_of_edges()} single-mutation links)')
    save(fig, 'fig2_genotype_graph.png', outdir)


# ── figure 3: NAND -> AND from the parallel start, N_P = 10^4 ─────────────────

def fig3(outdir, big):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axs[0, 0]; blank(ax)
    draw_parallel(ax)
    title(ax, 'A   starting network: $N$ parallel NAND$(I_1, I_2)$ gates')
    note(ax, 'output reads gate 1: computes NAND, fitness 0 for AND\n'
             'only 3 of the $2N+1$ edges are computing', loc=(0.02, 0.02), ha='left')

    ax = axs[0, 1]; loglog(ax, ylabel='mean total mutations to evolve AND, $\\langle M \\rangle$')
    r = series('par_1e4', 'M')
    xx = np.logspace(np.log10(4), np.log10(r[-1, 0] * 1.3), 200)
    ax.plot(xx, xx, color=MUTED, lw=1, ls=':', label='$y = x$ line')
    ax.plot(xx, [exact_M_parallel(n) for n in xx], color=PINK, lw=1.1, ls='-.',
            label='exact $\\langle M \\rangle$, analytic (not in paper)')
    dots(ax, r, BLUE, label='simulation (mean, 5–95%)')
    paper_fits(ax, r, BLUE, big, 1000)
    t = thirds(r, 1000)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, 'paper ($N \\leq 10^3$): $9.8\\,N^{0.96}$; 3 subsets: $A = 9.6 \\pm 3.7$, $b = 0.96 \\pm 0.11$\n'
             f'ours, 3 subsets of $N \\leq 10^3$: $A = {t[0]:.1f} \\pm {t[1]:.1f}$, '
             f'$b = {t[2]:.2f} \\pm {t[3]:.2f}$')
    title(ax, 'B   mutations grow linearly with $N$')

    ax = axs[1, 0]; loglog(ax, ylabel='mean mutations to evolve AND')
    rf = series('par_1e4', 'fix'); rl = series('par_1e4', 'lost'); rfb = series('par_1e4', 'fix_ben')
    dots(ax, rl, BLUE, label='non-fixating')
    dots(ax, rf, ORANGE, label='fixating (all)', marker='s')
    dots(ax, rfb, GREEN, label='fixating beneficial (not in paper)', marker='D', ms=5.5)
    fitline(ax, rl, BLUE, hi=1000, label_fmt='non-fixating, $N \\leq 10^3$: $y={A:.3g}\\,N^{{{b:.2f}}}$')
    ax.legend(loc='upper left', fontsize=9, title_fontsize=8.5, alignment='left',
              title='paper: non-fixating $9.5\\,N^{0.96}$; fixating ≈ const.\n'
                    'fixating neutral ≈ $\\langle M\\rangle / N_P$: grows once $N \\gtrsim N_P/10$')
    title(ax, 'C   fixating vs non-fixating mutations')

    ax = axs[1, 1]
    ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('mean fraction of non-fixating mutations')
    ax.grid(True, color=GRID_C, lw=0.6)
    keys = [('frac_neu', SKY, 'neutral', 'o'), ('frac_del', VERM, 'deleterious', 's'),
            ('frac_ben', GREEN, 'beneficial', 'D')]
    for key, c, lab, mk in keys:
        dots(ax, series('par_1e4', key), c, label=lab, marker=mk)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='center left', fontsize=9.5)
    ins = ax.inset_axes([0.50, 0.26, 0.46, 0.42])
    for key, c, lab, mk in keys[1:]:
        rr = series('par_1e4', key)
        ins.plot(rr[:, 0], rr[:, 1], mk, color=c, mec=INK, mew=0.5, ms=4.5)
    ins.set_xscale('log'); ins.set_yscale('log')
    ins.set_title('means on a log scale', fontsize=8.5, color=MUTED)
    ins.tick_params(labelsize=7.5); ins.grid(True, color=GRID_C, lw=0.5)
    title(ax, 'D   almost every lost mutation is neutral')
    save(fig, 'fig3_network_size.png', outdir)


# ── figure 4: origin of linear scaling ────────────────────────────────────────

def mech_panels(axs, exp, big, color, ce_ref=None, dloc=dict(loc='center right')):
    """Computing edges, f_b, and (if a third axis is given) P(c), P(b|c) panels."""
    ax = axs[0]; loglog(ax, ylabel='mean number of computing edges\n(averaged over residents)')
    r = series(exp, 'ce')
    if ce_ref is None:
        ax.set_yscale('linear'); ax.set_ylim(0, max(12, r[:, 3].max() * 1.15))
    else:
        xx = np.logspace(np.log10(r[0, 0]), np.log10(r[-1, 0]), 50)
        ax.plot(xx, ce_ref(xx), color='#bdbdbd', lw=5, solid_capstyle='round', zorder=1,
                label=ce_ref.__doc__)
    dots(ax, r, color, label='simulation (mean, 5–95%)')
    if ce_ref is not None:
        fit_lines(ax, r, color, big)
    ax.legend(loc='upper left', fontsize=9)

    ax = axs[1]; ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('mean fraction of beneficial mutations\nper computing edge, $f_b$')
    ax.grid(True, color=GRID_C, lw=0.6)
    r = series(exp, 'fb')
    dots(ax, r, color, label='simulation (mean, 5–95%)')
    ax.set_ylim(0, 1)
    f = fit(r)
    if f:
        note(ax, f'power-law fit exponent, all $N$: {num(f[1], "{:+.3f}")}', loc=(0.98, 0.95), va='top')
    ax.legend(loc='lower left', fontsize=9)

    if len(axs) < 3:
        return
    ax = axs[2]; loglog(ax, ylabel='probability')
    rc, rb = series(exp, 'pc'), series(exp, 'fb')
    dots(ax, rc, BLUE, label='$P(c)$ = computing edges / $(2N+1)$', bars=False)
    dots(ax, rb, GREEN, label='$P(b\\,|\\,c) = \\langle f_b \\rangle$', marker='D', bars=False)
    prod = rc.copy(); prod[:, 1] = rc[:, 1] * rb[:, 1]
    dots(ax, prod, PINK, label='$P(c)\\,P(b\\,|\\,c)$', marker='s', bars=False)
    txt = []
    for r_, lab in [(rc, 'P(c)'), (rb, 'P(b|c)'), (prod, 'product')]:
        f = fit(r_)
        if f:
            txt.append(f'{lab}: $N^{{{f[1]:+.2f}}}$')
            fitline(ax, r_, MUTED, label_fmt=None, lw=0.9, ls='--')
    note(ax, '   '.join(txt), loc=(0.02, 0.02), ha='left')
    ax.legend(fontsize=9, **dloc)


def draw_par_state(ax, edges, count):
    """Parallel-start network for Fig. 4A. edges: (src, dst, kind, new) with kind
    'c' (computing, blue) or 'n' (non-computing, black); new edges are dashed."""
    P = {'I1': (3.2, 8.7), 'I2': (6.4, 8.7), '1': (1.6, 5.4), '2': (4.2, 5.4), 'N': (8.6, 5.4),
         'O': (2.9, 2.2)}
    for k in ('I1', 'I2', 'O'):
        node(ax, P[k], {'I1': '$I_1$', 'I2': '$I_2$', 'O': '$O$'}[k], '#e3e3e3', w=1.3, h=0.9)
    for k in ('1', '2', 'N'):
        node(ax, P[k], k if k != 'N' else '$N$', w=1.1, h=0.9)
    ax.text(6.4, 5.4, '…', ha='center', va='center', fontsize=14)
    for a, b, kind, new in edges:
        c, lw, ls = SKY if kind == 'c' else INK, 2.0 if kind == 'c' else 1.0, '--' if new else '-'
        if P[a][1] == P[b][1]:                     # gate -> gate on the same row: side to side
            ax.add_patch(FancyArrowPatch((P[a][0] - 0.55, P[a][1]), (P[b][0] + 0.55, P[b][1]),
                                         arrowstyle='-|>', mutation_scale=12, color=c, lw=lw, ls=ls,
                                         shrinkA=1, shrinkB=1, zorder=4))
        else:
            dup = new and any(e[:2] == (a, b) and not e[3] for e in edges)
            arrow(ax, P[a], P[b], color=c, lw=lw, ls=ls, h=0.9, rad=-0.25 if dup else 0.0)
    ax.text(5.2, 2.2, f'computing edges = {count}', va='center', fontsize=10,
            color=BLUE if count else INK)


def draw_neighbourhood(ax, kinds, seed=1, arcs=True, weights=(0.25, 0.35, 0.25, 0.1, 0.05)):
    """Schematic local genotype graph (paper Figs 4D, 5D): one cluster per edge of the
    central network; kinds[i] = 'c' (computing edge: mixed fitness) or 'n' (neutral)."""
    from matplotlib.patches import Arc, Patch
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55); ax.set_aspect('equal'); ax.axis('off')
    rng = np.random.default_rng(seed)
    cats = list(FITC)
    k = len(kinds)
    for i, kd in enumerate(kinds):
        th0 = np.pi / 2 + 2 * np.pi * i / k
        for a in th0 + np.linspace(-0.27, 0.27, 6):
            x, y = 1.05 * np.cos(a), 1.05 * np.sin(a)
            f = cats[rng.choice(5, p=weights)] if kd == 'c' else 0.25
            ax.plot([0, x], [0, y], color=SKY if kd == 'c' else INK, lw=0.9, zorder=1)
            ax.scatter([x], [y], s=60, c=FITC[f], edgecolors=INK, linewidths=0.7, zorder=3)
    ax.scatter([0], [0], s=420, c=FITC[0.25], edgecolors=INK, linewidths=1, zorder=4)
    if arcs:
        ang = [np.degrees(np.pi / 2 + 2 * np.pi * i / k) for i in range(k)]
        cs = [a for a, kd in zip(ang, kinds) if kd == 'c']
        ns = [a for a, kd in zip(ang, kinds) if kd == 'n']
        if cs and ns:
            ax.add_patch(Arc((0, 0), 2.6, 2.6, theta1=min(cs) - 22, theta2=max(cs) + 22, color=SKY, lw=1.5))
            ax.add_patch(Arc((0, 0), 2.75, 2.75, theta1=max(cs) + 26, theta2=min(cs) - 26 + 360,
                             color=INK, lw=1.2))
            m = np.radians(np.mean(cs))
            ax.text(1.5 * np.cos(m), 1.5 * np.sin(m), '$O(N)$', color=BLUE, fontsize=12,
                    ha='center', va='center')
            ax.text(1.52, -0.05, '$O(N^2)$', fontsize=12, ha='left', va='center')
    ax.legend(handles=[Patch(fc=c, ec=INK, lw=0.6, label=f'{f:g}') for f, c in FITC.items()],
              title='fitness', fontsize=8.5, title_fontsize=9, loc='upper left',
              bbox_to_anchor=(0.97, 1.0), frameon=True, edgecolor='#cccccc')


def fig4(outdir, big):
    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.85, 1, 1])
    ax = fig.add_subplot(gs[:, 0]); blank(ax)
    title(ax, 'A   computing edges (blue) as mutations occur')
    S = [([('I1', '1', 'c', 0), ('I2', '1', 'c', 0), ('I1', '2', 'n', 0), ('I2', '2', 'n', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('1', 'O', 'c', 0)], 3),
         ([('I1', '1', 'c', 0), ('2', '1', 'c', 1), ('I1', '2', 'c', 0), ('I2', '2', 'c', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('1', 'O', 'c', 0)], 5),
         ([('I1', '1', 'n', 0), ('2', '1', 'n', 0), ('I1', '2', 'c', 0), ('I2', '2', 'c', 0),
           ('I1', 'N', 'n', 0), ('I2', 'N', 'n', 0), ('2', 'O', 'c', 1)], 3),
         ([('I1', '1', 'n', 0), ('2', '1', 'n', 0), ('I1', '2', 'c', 0), ('I2', '2', 'c', 0),
           ('I2', 'N', 'n', 1), ('I2', 'N', 'n', 0), ('2', 'O', 'c', 0)], 3)]
    steps = ['$I_2 \\to 1$ rewired to $2 \\to 1$', '$1 \\to O$ rewired to $2 \\to O$',
             '$I_1 \\to N$ rewired to $I_2 \\to N$ (non-computing)']
    h = 0.235
    for k, (edges, cnt) in enumerate(S):
        sub = ax.inset_axes([0.0, 1 - (k + 1) * h - k * 0.02, 1.0, h]); blank(sub)
        draw_par_state(sub, edges, cnt)
        if k < 3:
            yk = 1 - (k + 1) * h - k * 0.02 - 0.005
            ax.annotate('', xy=(0.12, yk - 0.012), xytext=(0.12, yk + 0.012), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
            ax.text(0.17, yk, steps[k], transform=ax.transAxes, fontsize=9, va='center', color=MUTED)
    note(ax, 'new edges dashed; computing = on a path from the inputs to $O$', loc=(0.5, -0.02),
         ha='center', va='top')

    axB, axC = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    axD, axE = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    mech_panels([axB, axC, axE], 'par_1e4', big, BLUE, dloc=dict(loc='lower left', bbox_to_anchor=(0, 0.07)))
    title(axB, 'B   computing edges stay ≈ constant')
    title(axC, 'C   $f_b$ is roughly independent of $N$')
    draw_neighbourhood(axD, ['n', 'c', 'c', 'n', 'n', 'n'], seed=3)
    title(axD, 'D   local neighbourhood of a genotype')
    title(axE, 'E   so $P(b) \\propto P(c) P(b|c) \\sim 1/N$ (not in paper)')
    save(fig, 'fig4_mechanism.png', outdir)


def dominant_runs(exp, rows, share=0.25):
    """(N, M) of single runs that make up more than `share` of all mutations at that N."""
    out = []
    for N in rows[:, 0]:
        M = load(exp, int(N))['M']
        if M.max() > share * M.sum():
            out.append((int(N), M.max()))
    return out


# ── figure 5: chain start ─────────────────────────────────────────────────────

def fig5(outdir, big):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    ax = axs[0, 0]; blank(ax)
    draw_chain(ax)
    title(ax, 'A   starting network: a chain (every edge computing)')
    note(ax, 'gate 1 = NAND$(I_1, I_2)$, gates 2…N = NOT of the previous\n'
             'odd $N$: computes NAND, fitness 0 for AND', loc=(0.02, 0.0), ha='left')

    def ref(n):
        """$y = 2N$ (paper's fit)"""
        return 2 * n
    mech_panels([axs[0, 1], axs[0, 2]], 'chain_1e4', big, VERM, ce_ref=ref)
    h = halves(series('chain_1e4', 'ce', nmax=601))
    title(axs[0, 1], 'B   computing edges grow as $2N$')
    note(axs[0, 1], f'paper: $y = 2N^{{1.0}}$, halves 0.99 / 1\nours ($N \\leq 601$): halves '
                    f'{h[0]:.2f} / {h[1]:.2f}')
    title(axs[0, 2], 'C   $f_b$ levels off for $N > 20$')

    draw_neighbourhood(axs[1, 0], ['c'] * 6, seed=5, arcs=False, weights=(0.3, 0.3, 0.25, 0.1, 0.05))
    axs[1, 0].text(0, -1.5, 'every mutation hits a computing edge: $O(N^2)$ such neighbours',
                   ha='center', fontsize=9, color=MUTED)
    title(axs[1, 0], 'D   local neighbourhood of the chain')

    ax = axs[1, 1]; loglog(ax, ylabel='mean total mutations to evolve AND')
    r = series('chain_1e4', 'M'); rm = series('chain_1e4', 'M', stat='median')
    dots(ax, r, VERM, label='mean (5–95%)')
    dots(ax, rm, VERM, label='median (not in paper)', hollow=True, marker='s', bars=False, offset=1.12)
    paper_fits(ax, r, VERM, big, 601)
    h = halves(r[r[:, 0] <= 601])
    ax.legend(loc='upper left', fontsize=9)
    txt = (f'paper ($N \\leq 601$): $31\\,N^{{0.07}}$, halves 0.07 / −0.09\n'
           f'ours ($N \\leq 601$): halves {num(h[0])} / {num(h[1])}')
    for N, Mx in dominant_runs('chain_1e4', r):
        txt += f'\n$N = {N}$: mean set by one run with $M$ = {Mx:,.0f}'
    note(ax, txt)
    title(ax, 'E   mutations do not grow with $N$')

    ax = axs[1, 2]; loglog(ax, ylabel='$\\langle M \\rangle \\times P(c)\\,P(b|c)$')
    for exp, c, lab, mk in [('par_1e4', BLUE, 'parallel start (Fig. 3)', 'o'),
                            ('chain_1e4', VERM, 'chain start (Fig. 5)', 's')]:
        rM, rc, rb = series(exp, 'M'), series(exp, 'pc'), series(exp, 'fb')
        n = min(len(rM), len(rc), len(rb))
        y = rM[:n].copy(); y[:, 1] = rM[:n, 1] * rc[:n, 1] * rb[:n, 1]
        dots(ax, y, c, label=lab, marker=mk, bars=False)
    ax.set_ylim(1, 300); ax.keep_ylim = True
    ax.legend(loc='upper left', fontsize=9)
    note(ax, 'Eq. 2: $\\langle M\\rangle \\cdot P(c)P(b|c) \\approx$ (steps) / $P_{fix}$\n'
             '(product of means); flat in $N$ for both starts')
    title(ax, 'F   Eq. 2 accounts for both scalings (not in paper)')
    save(fig, 'fig5_chain_start.png', outdir)


# ── supplementary figures ─────────────────────────────────────────────────────

def figS1(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axs[0]; loglog(ax, ylabel='mean number of mutations to reach AND')
    r = series('rw', 'M')
    dots(ax, series('par_1e4', 'M', nmax=r[-1, 0] if len(r) else None), BLUE,
         label='with selection, $N_P = 10^4$ (Fig. 3B; not in paper)', bars=False, hollow=True,
         offset=1.15, ms=5.5)
    dots(ax, r, PURP, label='random walk (mean, 5–95%)')
    paper_fits(ax, r, PURP, big, 1000)
    t = thirds(r, 1000)
    ax.legend(loc='upper left', fontsize=8.8)
    note(ax, 'paper ($N \\leq 10^3$): $18.7\\,N^{0.8}$; 3 subsets: $b = 0.8 \\pm 0.2$\n'
             f'ours, 3 subsets of $N \\leq 10^3$: $b = {t[2]:.2f} \\pm {t[3]:.2f}$')
    title(ax, 'A   random walk from the parallel start')
    ax = axs[1]; loglog(ax, ylabel='computing edges per network visited')
    r = series('rw', 'ce')
    dots(ax, r, PURP, label='random walk (mean, 5–95%;\nlower bars = minimal module, 3–5 edges)')
    fitline(ax, r, PURP)
    dots(ax, series('par_1e4', 'ce', nmax=r[-1, 0] if len(r) else None), BLUE,
         label='with selection, $N_P = 10^4$', bars=False, hollow=True)
    ax.legend(loc='upper left', fontsize=9)
    title(ax, 'B   without selection the module grows (not in paper)')
    save(fig, 'figS1_random_walk.png', outdir)


def figS2(outdir, big):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, N, lab in [(axs[0, 0], 40, 'A'), (axs[0, 1], 400, 'B')]:
        d = load('par_1e4', N)
        M = d['M']
        top = np.percentile(M, 99.5)
        ax.hist(M[M <= top], bins=np.linspace(0, top, 30), color=BLUE, alpha=0.85, ec='white', lw=0.5)
        ax.axvline(M.mean(), color=INK, lw=1.4, label=f'mean {M.mean():.0f}')
        ax.axvline(np.median(M), color=INK, lw=1.4, ls='--', label=f'median {np.median(M):.0f}')
        ax.set_xlim(0, top)
        ax.set_xlabel('number of mutations to evolve AND'); ax.set_ylabel('number of simulations')
        ax.legend(fontsize=9.5)
        note(ax, f'{(M > top).sum()} runs beyond the axis (max {M.max():,.0f})', loc=(0.98, 0.70), va='top')
        title(ax, f'{lab}   distribution at $N = {N}$ ({len(M)} runs)')
    ax = axs[1, 0]
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$M / \\langle M \\rangle$'); ax.set_ylabel('fraction of runs with $M$ at least this large')
    ax.grid(True, color=GRID_C, lw=0.6)
    Ns = [n for n in EXPS['par_1e4'][3] if n <= NMAX and load('par_1e4', n) is not None]
    cm = plt.get_cmap('viridis')
    for i, N in enumerate(Ns):
        M = np.sort(load('par_1e4', N)['M'])
        surv = 1 - np.arange(len(M)) / len(M)
        ax.plot(M / M.mean(), surv, color=cm(i / max(len(Ns) - 1, 1)), lw=1.2,
                label=f'$N={N:,}$' if i in (0, len(Ns) // 2, len(Ns) - 1) else None)
    x = np.logspace(-2, np.log10(7), 50)
    ax.plot(x, np.exp(-x), color=INK, ls=':', lw=1.2, label='exponential $e^{-x}$')
    ax.set_ylim(1e-3, 1.3); ax.set_xlim(1e-2, 15); ax.keep_ylim = ax.keep_xlim = True
    ax.legend(fontsize=9)
    title(ax, 'C   one shape at every $N$ (not in paper)')
    ax = axs[1, 1]; loglog(ax, ylabel='median number of mutations to evolve AND')
    r = series('par_1e4', 'M', stat='median')
    dots(ax, r, BLUE, label='median (5–95% of runs)')
    paper_fits(ax, r, BLUE, big, 1000)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, 'paper ($N \\leq 10^3$): $7.8\\,N^{0.96}$, no error bars')
    title(ax, 'D   median grows like the mean')
    save(fig, 'figS2_distributions.png', outdir)


def figS3(outdir, big):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8))
    for k, (ax, (exp, lab, paper)) in enumerate(zip(axs, [
            ('andnot_10', '$I_1$ AND NOT $I_2$', '$107.8\\,N^{0.82}$ [0.64, 0.95]'),
            ('or_10', 'OR', '$64.6\\,N^{0.86}$ [0.78, 0.94]'),
            ('true_10', 'TRUE', '$34.8\\,N^{0.91}$ [0.86, 0.94]')])):
        loglog(ax, ylabel=f'mean number of mutations to evolve {lab}')
        r = series(exp, 'M')
        dots(ax, r, PINK, label='simulation (mean, 5–95%)')
        paper_fits(ax, r, PINK, big, 1000)
        rp = r[r[:, 0] <= 1000]
        txt = f'paper ($N \\leq 10^3$): {paper}'
        if len(rp) >= 7:
            txt += (f'\nours, first 3 / last 4 points of $N \\leq 10^3$: '
                    f'[{fit(rp[:3])[1]:.2f}, {fit(rp[-4:])[1]:.2f}]')
        note(ax, txt)
        ax.legend(loc='upper left', fontsize=9)
        title(ax, f'{"ABC"[k]}   NAND → {lab}, $N_P = 10$')
    save(fig, 'figS3_other_targets.png', outdir)


def small_pop(outdir, big, exp, exp_big, color, name, letter_titles, paper_notes, paper_hi,
              subsets, pc_linear=False):
    """Figs S4 / S5. paper_hi: the paper's largest N; subsets: its two N ranges for
    the sensitivity exponents; paper_notes: its (overall, subset) numbers per panel."""
    def compare(r, paper):
        b = b_on(r, 0, paper_hi)
        lo, hi = (b_on(r, *rg) for rg in subsets)
        return (f'paper: {paper}\nours ($N \\leq {paper_hi}$): {num(b)} '
                f'({num(lo)} / {num(hi)})')

    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axs[0, 0]; loglog(ax, ylabel='mean total mutations to evolve AND')
    r = series(exp, 'M')
    dots(ax, r, color, label='$N_P = 10$ (mean, 5–95%)')
    if pc_linear:
        dots(ax, series(exp, 'M', stat='median'), color, label='median (not in paper)', hollow=True,
             marker='s', bars=False, offset=1.12)
    paper_fits(ax, r, color, big, paper_hi)
    note(ax, compare(r, paper_notes[0]))
    ax.legend(loc='upper left', fontsize=9)
    title(ax, letter_titles[0])

    ax = axs[0, 1]; loglog(ax, ylabel='mean number of fixating mutations')
    rfx = series(exp, 'fix')
    dots(ax, rfx, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'fix', nmax=rfx[-1, 0] if len(rfx) else None), INK,
         label='$N_P = 10^4$', hollow=True, marker='s')
    dots(ax, series(exp, 'fix_ben'), PURP, label='$N_P = 10$, beneficial only (not in paper)',
         marker='D', ms=4.5, bars=False, offset=1.15)
    ax.legend(loc='upper left', fontsize=9)
    title(ax, letter_titles[1])

    ax = axs[1, 0]
    r = series(exp, 'pc')
    if pc_linear:
        ax.set_xscale('log'); ax.set_xlabel('network size $N$'); ax.set_ylabel('$P(c)$')
        ax.grid(True, color=GRID_C, lw=0.6); ax.set_ylim(0, 1.05)
    else:
        loglog(ax, ylabel='$P(c)$')
    dots(ax, r, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'pc', nmax=r[-1, 0] if len(r) else None), INK,
         label='$N_P = 10^4$ (not in paper)', hollow=True, marker='s', bars=False)
    paper_fits(ax, r, color, big, paper_hi)
    note(ax, compare(r, paper_notes[1]), loc=(0.02, 0.04), ha='left')
    ax.legend(loc='upper right' if not pc_linear else 'lower right',
              bbox_to_anchor=None if not pc_linear else (1.0, 0.12), fontsize=9)
    title(ax, letter_titles[2])

    ax = axs[1, 1]; ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('$P(b\\,|\\,c) = \\langle f_b \\rangle$'); ax.grid(True, color=GRID_C, lw=0.6)
    r = series(exp, 'fb')
    dots(ax, r, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'fb', nmax=r[-1, 0] if len(r) else None), INK,
         label='$N_P = 10^4$ (not in paper)', hollow=True, marker='s', bars=False)
    paper_fits(ax, r, color, big, paper_hi)
    note(ax, compare(r, paper_notes[2]), loc=(0.98, 0.95), va='top')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left', fontsize=8.5)
    title(ax, letter_titles[3])
    save(fig, name, outdir)


def figS4(outdir, big):
    small_pop(outdir, big, 'par_10', 'par_1e4', ORANGE, 'figS4_small_population.png',
              ['A   $N_P = 10$: total mutations', 'B   fixating mutations',
               'C   $P(c)$ falls, then levels off', 'D   $P(b|c)$ nearly flat'],
              ['0.90 (0.87 / 0.92)', '−0.85 (−0.81 / −0.87)', '−0.020 (−0.049 / −0.012)'],
              600, [(5, 40), (60, 600)])


def figS5(outdir, big):
    small_pop(outdir, big, 'chain_10', 'chain_1e4', GREEN, 'figS5_small_population_chain.png',
              ['A   chain start, $N_P = 10$: total mutations', 'B   fixating mutations',
               'C   $P(c) \\approx 1$', 'D   $P(b|c)$'],
              ['$41.3\\,N^{0.10}$ (0.21 / −0.17)', '0.027 (0.046 / 0.014)', '−0.135 (−0.238 / −0.036)'],
              501, [(5, 41), (81, 501)], pc_linear=True)


def figS6(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, exp, c, lab, lin in [(axs[0], 'par_1e4', BLUE, 'A   parallel start: time is flat', True),
                                 (axs[1], 'chain_1e4', VERM, 'B   chain start: time falls ≈ as $1/N$', False)]:
        loglog(ax, ylabel='mean time to evolve AND')
        r = series(exp, 'T')
        dots(ax, r, c, label='simulation (mean, 5–95%)')
        f = fit(r)
        fitline(ax, r, MUTED, ls='--', lw=1.0,
                label_fmt='fit, all $N$ (not in paper): $y={A:.3g}\\,N^{{{b:.2f}}}$')
        if lin:
            ax.set_yscale('linear'); ax.set_ylim(0, 2)
            ax.legend(loc='upper right', fontsize=9)
            note(ax, 'time drawn after the run:\n$T \\mid M \\sim$ Gamma$(M,\\ \\tau / (N_P (2N+1)))$, $\\tau = 1000$;\n'
                     "same distribution as the Methods' exponential waits", loc=(0.02, 0.80),
                 ha='left', va='top')
        else:
            ax.legend(loc='lower left', fontsize=9)
        title(ax, lab)
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
            print(f'skipped {name}: {type(e).__name__}: {e}')
            plt.close('all')


if __name__ == '__main__':
    main()
