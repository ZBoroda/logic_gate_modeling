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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_replication.run import EXPS, path, AND, ANDNOT, OR, TRUE
from bottleneck_replication.theory import exact_M_parallel

HERE = os.path.dirname(__file__)
INK, MUTED, GRID_C = '#1a1a1a', '#6b6b6b', '#e6e6e6'
BLUE, ORANGE, GREEN, PINK, SKY, VERM, YEL, PURP = ('#0072B2', '#E69F00', '#009E73', '#CC79A7',
                                                  '#56B4E9', '#D55E00', '#F0E442', '#2E2A7A')
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
    out['pc'] = out['ce'] / (2 * N + 1)
    out['frac_ben'] = out['lost_ben'] / out['lost']
    out['frac_del'] = out['lost_del'] / out['lost']
    out['frac_neu'] = out['lost_neu'] / out['lost']
    Np = EXPS[exp][2] or 1.0
    rng = np.random.default_rng(N)
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


def fit(r, lo=0, hi=np.inf):
    """y = A N^b on rows with lo <= N <= hi. Returns (A, b, b_se)."""
    m = (r[:, 0] >= lo) & (r[:, 0] <= hi) & (r[:, 1] > 0)
    if m.sum() < 3:
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
    if ax.get_xscale() == 'log':
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


def fig1(outdir):
    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])
    # A: example network NAND gates g1 = NAND(I1, I2), g2 = NAND(I1, g1), O <- g2
    ax = fig.add_subplot(gs[0]); blank(ax)
    title(ax, 'A   a network and its function')
    I1, I2, g1, g2, O = (2.0, 9.0), (4.6, 9.0), (4.0, 6.8), (2.4, 4.6), (2.4, 2.4)
    for p, l, f in [(I1, '$I_1$', '#e3e3e3'), (I2, '$I_2$', '#e3e3e3'), (g1, '1', 'white'),
                    (g2, '2', 'white'), (O, '$O$', '#e3e3e3')]:
        node(ax, p, l, f)
    arrow(ax, I1, g1); arrow(ax, I2, g1); arrow(ax, I1, g2); arrow(ax, g1, g2); arrow(ax, g2, O)
    v1 = (~(0b1100 & 0b1010)) & 0xF
    v2 = (~(0b1100 & v1)) & 0xF
    rows = [('0', '0'), ('0', '1'), ('1', '0'), ('1', '1')]
    out, tgt = tt_str(v2), tt_str(AND)
    tx, ty = 6.0, 7.2
    for j, h in enumerate(['$I_1$', '$I_2$', '$O$', 'AND']):
        ax.text(tx + 0.9 * j, ty + 0.9, h, fontsize=10, fontweight='bold', ha='center')
    for i, (a, b) in enumerate(rows):
        ok = out[i] == tgt[i]
        for j, v in enumerate([a, b, out[i], tgt[i]]):
            ax.text(tx + 0.9 * j, ty - 0.75 * i, v, fontsize=10, family='monospace', ha='center')
        ax.text(tx + 3.55, ty - 0.75 * i, '✓' if ok else '✗', color=GREEN if ok else VERM,
                fontsize=11, ha='center')
    f = sum(o == t for o, t in zip(out, tgt)) / 4
    ax.text(tx, ty - 3.6, f'fitness for AND = {f:g}', fontsize=10.5)
    ax.text(0.2, 0.6, 'every gate is NAND; edges point from source to gate', fontsize=9,
            color=MUTED)

    # B: a beneficial rewiring found by search from the same network
    ax = fig.add_subplot(gs[1]); blank(ax)
    title(ax, 'B   one mutation rewires one edge')
    node(ax, I1, '$I_1$', '#e3e3e3'); node(ax, I2, '$I_2$', '#e3e3e3')
    node(ax, g1, '1'); node(ax, g2, '2'); node(ax, O, '$O$', '#e3e3e3')
    # mutation: input I1 -> 2 is rewired to 1 -> 2: g2 = NAND(g1, g1) = AND
    arrow(ax, I1, g1); arrow(ax, I2, g1); arrow(ax, g1, g2); arrow(ax, g2, O)
    arrow(ax, I1, g2, color=VERM, lw=2, ls='--')
    arrow(ax, g1, g2, color=GREEN, lw=2.2, rad=-0.35)
    v2b = (~(v1 & v1)) & 0xF
    fb = sum(o == t for o, t in zip(tt_str(v2b), tgt)) / 4
    ax.text(5.6, 5.4, 'removed: $I_1 \\to 2$', color=VERM, fontsize=10)
    ax.text(5.6, 4.7, 'added:    $1 \\to 2$', color=GREEN, fontsize=10)
    ax.text(5.6, 3.6, f'fitness for AND {f:g} → {fb:g}\n(gate 2 now computes AND)', fontsize=10.5)
    ax.text(0.2, 0.6, 'new source: any node except the old one, the gate itself and $O$;\n'
                      'proposals that close a cycle are redrawn', fontsize=9, color=MUTED)

    # C: flowchart
    ax = fig.add_subplot(gs[2]); blank(ax)
    title(ax, 'C   evolutionary loop (SSWM, Moran)')
    steps = [('resident network', 8.6), ('propose one mutation\n(uniform edge, new source)', 6.9),
             ('fixes with probability\n$\\pi(s) = (1-e^{-s})/(1-e^{-N_P s})$', 5.0),
             ('fixed: mutant becomes resident\nlost: resident unchanged', 3.1),
             ('stop when fitness = 1', 1.3)]
    for s, y in steps:
        ax.add_patch(FancyBboxPatch((1.0, y - 0.65), 8.0, 1.3, boxstyle='round,pad=0.02,rounding_size=0.2',
                                    fc='#f4f7fb' if y != 1.3 else '#fdf3e3', ec='#8aa1b8', lw=1.1))
        ax.text(5.0, y, s, ha='center', va='center', fontsize=9.8)
    for (_, y0), (_, y1) in zip(steps[:-1], steps[1:]):
        ax.annotate('', xy=(5.0, y1 + 0.67), xytext=(5.0, y0 - 0.67),
                    arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.1))
    ax.annotate('', xy=(9.0, 8.6), xytext=(9.0, 3.1),
                arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1.1,
                                connectionstyle='arc3,rad=-0.5'))
    ax.text(9.95, 5.85, 'repeat', rotation=90, ha='center', va='center', fontsize=9, color=MUTED)
    ax.text(5.0, -0.1, '$M$ counts every proposed mutation, fixed or lost', ha='center',
            fontsize=9, color=MUTED)
    save(fig, 'fig1_model.png', outdir)


# ── figure 2: genotype graph of size-2 networks ───────────────────────────────

def fig2(outdir):
    import networkx as nx
    from faster_circuit.core import fitness
    from faster_circuit import has_cycle_via
    from goals import and_funct
    N = 2
    srcs = {0: [0, 1, 3], 1: [0, 1, 2]}
    genomes = []
    for a, b, c, d, o in itertools.product(srcs[0], srcs[0], srcs[1], srcs[1], [2, 3]):
        g = [a, b, c, d, o]
        if (2 in (c, d)) and (3 in (a, b)):
            continue                                        # 2 <-> 3 cycle
        genomes.append(tuple(g))
    idx = {g: i for i, g in enumerate(genomes)}
    G = nx.Graph()
    G.add_nodes_from(range(len(genomes)))
    for g in genomes:
        for e in range(5):
            for u in (range(2, 4) if e == 4 else range(4)):
                if u == g[e] or (e < 4 and u == e // 2 + 2):
                    continue
                h = list(g); h[e] = u; h = tuple(h)
                if h in idx:
                    G.add_edge(idx[g], idx[h])
    fit_ = np.array([fitness(list(g), and_funct) for g in genomes])
    pos = nx.spring_layout(G, seed=4, k=0.35, iterations=300)
    cmap = {0.0: PURP, 0.25: BLUE, 0.5: GREEN, 0.75: ORANGE, 1.0: YEL}
    fig, ax = plt.subplots(figsize=(9.5, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#c8c8c8', width=0.5)
    for f, c in cmap.items():
        nodes = [i for i in G if fit_[i] == f]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=c, node_size=70,
                               edgecolors=INK, linewidths=0.5, ax=ax,
                               label=f'fitness {f:g}  ({len(nodes)} genotypes)')
    start = idx[(0, 1, 0, 1, 2)]
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='none', node_size=260,
                           edgecolors=VERM, linewidths=2.2, ax=ax)
    x, y = pos[start]
    ax.annotate('parallel start\n(NAND, fitness 0)', (x, y), xytext=(x + 0.25, y + 0.22),
                fontsize=9.5, color=VERM, arrowprops=dict(arrowstyle='-', color=VERM, lw=1))
    ax.axis('off')
    ax.legend(loc='upper left', fontsize=9.5, title='target AND', title_fontsize=10)
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

    ax = axs[0, 1]; loglog(ax, ylabel='total mutations to AND, $\\langle M \\rangle$')
    r = series('par_1e4', 'M')
    xx = np.logspace(np.log10(4), np.log10(r[-1, 0] * 1.3), 200)
    ax.plot(xx, xx, color=MUTED, lw=1, ls=':', label='$y = N$')
    ax.plot(xx, [exact_M_parallel(n) for n in xx], color=INK, lw=1, alpha=0.6,
            label='exact $\\langle M \\rangle$ (bottleneck note, Eq. 3)')
    dots(ax, r, BLUE, label='simulation (mean, 5–95%)')
    f = fit_lines(ax, r, BLUE, big)
    h = halves(r)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, f'paper ($N \\leq 10^3$): $9.8\\,N^{{0.96}}$\n'
             f'lower / upper half of points: $b$ = {h[0]:.2f} / {h[1]:.2f}')
    title(ax, 'B   mutations grow linearly with $N$')

    ax = axs[1, 0]; loglog(ax, ylabel='mutations per run')
    rf = series('par_1e4', 'fix'); rl = series('par_1e4', 'lost'); rfb = series('par_1e4', 'fix_ben')
    dots(ax, rl, BLUE, label='non-fixating')
    dots(ax, rf, ORANGE, label='fixating (all)', marker='s')
    dots(ax, rfb, GREEN, label='fixating beneficial', marker='D', ms=5.5)
    fitline(ax, rl, BLUE, label_fmt='non-fixating fit $y={A:.3g}\\,N^{{{b:.2f}}}$')
    ax.legend(loc='upper left', fontsize=9)
    note(ax, 'paper: non-fixating $9.5\\,N^{0.96}$; fixating ≈ const.\n'
             'fixating neutral mutations ≈ $\\langle M\\rangle / N_P$ grow once $N \\gtrsim N_P/10$',
         loc=(0.02, 0.70), ha='left', va='top')
    title(ax, 'C   fixating vs non-fixating mutations')

    ax = axs[1, 1]
    ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('fraction of non-fixating mutations')
    ax.grid(True, color=GRID_C, lw=0.6)
    for key, c, lab, mk in [('frac_neu', SKY, 'neutral', 'o'), ('frac_del', VERM, 'deleterious', 's'),
                            ('frac_ben', GREEN, 'beneficial', 'D')]:
        dots(ax, series('par_1e4', key), c, label=lab, marker=mk)
    ax.set_yscale('symlog', linthresh=1e-5)
    ax.set_ylim(0, 1.6)
    ax.legend(loc='center right', fontsize=9.5)
    title(ax, 'D   almost every lost mutation is neutral')
    save(fig, 'fig3_network_size.png', outdir)


# ── figure 4: origin of linear scaling ────────────────────────────────────────

def mech_panels(axs, exp, big, color, ce_ref=None, dloc=dict(loc='center right')):
    """Computing edges, f_b, and P(c), P(b|c) panels for one experiment."""
    ax = axs[0]; loglog(ax, ylabel='computing edges per resident')
    r = series(exp, 'ce')
    if ce_ref is None:
        ax.set_yscale('linear'); ax.set_ylim(0, max(12, r[:, 3].max() * 1.15))
    dots(ax, r, color, label='simulation (mean, 5–95%)')
    fit_lines(ax, r, color, big)
    if ce_ref:
        xx = np.logspace(np.log10(r[0, 0]), np.log10(r[-1, 0]), 50)
        ax.plot(xx, ce_ref(xx), color=MUTED, lw=1, ls=':', label=ce_ref.__doc__)
    ax.legend(loc='upper left', fontsize=9)

    ax = axs[1]; ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('$f_b$: fraction of beneficial rewirings\nper computing edge')
    ax.grid(True, color=GRID_C, lw=0.6)
    r = series(exp, 'fb')
    dots(ax, r, color, label='simulation (mean, 5–95%)')
    ax.set_ylim(0, max(0.6, r[:, 3].max() * 1.1))
    f = fit(r)
    if f:
        note(ax, f'power-law fit exponent: {f[1]:+.3f} ± {f[2]:.3f}', loc=(0.98, 0.95), va='top')
    ax.legend(loc='lower left', fontsize=9)

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


def fig4(outdir, big):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axs[0, 0]; blank(ax, (0, 20), (0, 10))
    title(ax, 'A   computing edges (blue) before and after a mutation')
    sub1 = ax.inset_axes([0.0, 0.08, 0.48, 0.88]); blank(sub1); draw_parallel(sub1, computing=BLUE)
    sub2 = ax.inset_axes([0.52, 0.08, 0.48, 0.88]); blank(sub2)
    draw_parallel(sub2, computing=BLUE, rewired=True)
    sub1.text(5, 0.6, '3 computing edges', ha='center', fontsize=10, color=BLUE)
    sub2.text(5, 0.6, '$I_1 \\to 1$ rewired to $2 \\to 1$: 5', ha='center', fontsize=10, color=BLUE)
    note(ax, 'computing edge: on a path from the inputs to $O$', loc=(0.5, 0.0), ha='center')

    mech_panels([axs[0, 1], axs[1, 0], axs[1, 1]], 'par_1e4', big, BLUE,
                dloc=dict(loc='lower left', bbox_to_anchor=(0, 0.07)))
    title(axs[0, 1], 'B   computing edges stay ≈ constant')
    title(axs[1, 0], 'C   $f_b$ is roughly independent of $N$')
    title(axs[1, 1], 'D   so $P(b) \\propto P(c) P(b|c) \\sim 1/N$')
    save(fig, 'fig4_mechanism.png', outdir)


# ── figure 5: chain start ─────────────────────────────────────────────────────

def fig5(outdir, big):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    ax = axs[0, 0]; blank(ax)
    draw_chain(ax)
    title(ax, 'A   starting network: a chain (every edge computing)')
    note(ax, 'gate 1 = NAND$(I_1, I_2)$, gates 2…N = NOT of the previous\n'
             'odd $N$: computes NAND, fitness 0 for AND', loc=(0.02, 0.0), ha='left')

    def ref(n):
        """$y = 2N$"""
        return 2 * n
    mech_panels([axs[0, 1], axs[0, 2], axs[1, 0]], 'chain_1e4', big, VERM, ce_ref=ref)
    title(axs[0, 1], 'B   computing edges grow as $2N$')
    note(axs[0, 1], 'paper: $y = 2N^{1.0}$')
    title(axs[0, 2], 'C   $f_b$ levels off for $N > 20$')
    title(axs[1, 0], 'D   $P(c) \\approx 1$: $P(b)$ independent of $N$')

    ax = axs[1, 1]; loglog(ax, ylabel='total mutations to AND')
    r = series('chain_1e4', 'M'); rm = series('chain_1e4', 'M', stat='median')
    dots(ax, r, VERM, label='mean (5–95%)')
    dots(ax, rm, VERM, label='median', hollow=True, marker='s', bars=False, offset=1.12)
    fit_lines(ax, r, VERM, big)
    h = halves(r)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, f'paper ($N \\leq 501$): $31\\,N^{{0.07}}$\nhalves: $b$ = {h[0]:.2f} / {h[1]:.2f}')
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
    note(ax, 'Eq. 2: $\\langle M\\rangle \\approx$ (steps) / $(P(c) P(b|c) P_{fix})$\n'
             'flat in $N$ for both starts: the scaling of $\\langle M\\rangle$\n'
             'is the scaling of $1 / P(c)P(b|c)$')
    title(ax, 'F   Eq. 2 accounts for both scalings')
    save(fig, 'fig5_chain_start.png', outdir)


# ── supplementary figures ─────────────────────────────────────────────────────

def figS1(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axs[0]; loglog(ax, ylabel='mutations to AND')
    r = series('rw', 'M')
    dots(ax, r, PURP, label='random walk (every mutation accepted)')
    fit_lines(ax, r, PURP, big)
    dots(ax, series('par_1e4', 'M', nmax=r[-1, 0] if len(r) else None), BLUE,
         label='with selection, $N_P = 10^4$ (Fig. 3B)', bars=False, hollow=True)
    h = halves(r)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, f'paper: $18.7\\,N^{{0.8}}$\nhalves: $b$ = {h[0]:.2f} / {h[1]:.2f}' if h else 'paper: $18.7\\,N^{0.8}$')
    title(ax, 'A   random walk from the parallel start')
    ax = axs[1]; loglog(ax, ylabel='computing edges per network visited')
    r = series('rw', 'ce')
    dots(ax, r, PURP, label='random walk')
    fitline(ax, r, PURP)
    dots(ax, series('par_1e4', 'ce', nmax=r[-1, 0] if len(r) else None), BLUE,
         label='with selection, $N_P = 10^4$', bars=False, hollow=True)
    ax.legend(loc='upper left', fontsize=9)
    title(ax, 'B   without selection the computing module grows')
    save(fig, 'figS1_random_walk.png', outdir)


def figS2(outdir, big):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    for ax, N, lab in [(axs[0, 0], 40, 'A'), (axs[0, 1], 400, 'B')]:
        d = load('par_1e4', N)
        M = d['M']
        ax.hist(M, bins=40, color=BLUE, alpha=0.85, ec='white', lw=0.5)
        ax.axvline(M.mean(), color=INK, lw=1.4, label=f'mean {M.mean():.0f}')
        ax.axvline(np.median(M), color=INK, lw=1.4, ls='--', label=f'median {np.median(M):.0f}')
        ax.set_xlabel('total mutations to AND'); ax.set_ylabel('runs')
        ax.legend(fontsize=9.5)
        title(ax, f'{lab}   distribution at $N = {N}$ ({len(M)} runs)')
    ax = axs[1, 0]
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$M / \\langle M \\rangle$'); ax.set_ylabel('fraction of runs with larger $M$')
    ax.grid(True, color=GRID_C, lw=0.6)
    Ns = [n for n in EXPS['par_1e4'][3] if n <= NMAX and load('par_1e4', n) is not None]
    cm = plt.get_cmap('viridis')
    for i, N in enumerate(Ns):
        M = np.sort(load('par_1e4', N)['M'])
        surv = 1 - np.arange(len(M)) / len(M)
        ax.plot(M / M.mean(), surv, color=cm(i / max(len(Ns) - 1, 1)), lw=1.2,
                label=f'$N={N:,}$' if i in (0, len(Ns) // 2, len(Ns) - 1) else None)
    x = np.logspace(-2, 1, 50)
    ax.plot(x, np.exp(-x), color=INK, ls=':', lw=1.2, label='exponential $e^{-x}$')
    ax.set_ylim(1e-3, 1.3); ax.set_xlim(1e-2, 15)
    ax.legend(fontsize=9)
    title(ax, 'C   one shape at every $N$ (rescaled survival curves)')
    ax = axs[1, 1]; loglog(ax, ylabel='median total mutations to AND')
    r = series('par_1e4', 'M', stat='median')
    dots(ax, r, BLUE, label='median (5–95%)')
    fit_lines(ax, r, BLUE, big)
    ax.legend(loc='upper left', fontsize=9)
    note(ax, 'paper: $7.8\\,N^{0.96}$')
    title(ax, 'D   median grows like the mean')
    save(fig, 'figS2_distributions.png', outdir)


def figS3(outdir, big):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (exp, lab, paper) in zip(axs, [
            ('andnot_10', '$I_1$ AND NOT $I_2$', '$107.8\\,N^{0.82}$ [0.64, 0.95]'),
            ('or_10', 'OR', '$64.6\\,N^{0.86}$ [0.78, 0.94]'),
            ('true_10', 'TRUE', '$34.8\\,N^{0.91}$ [0.86, 0.94]')]):
        loglog(ax, ylabel=f'mutations to {lab}')
        r = series(exp, 'M')
        dots(ax, r, PINK, label='simulation (mean, 5–95%)')
        fit_lines(ax, r, PINK, big)
        txt = f'paper: {paper}'
        if len(r) >= 7:
            txt += f'\nfirst 3 / last 4 points: $b$ = {fit(r[:3])[1]:.2f} / {fit(r[-4:])[1]:.2f}'
        note(ax, txt)
        ax.legend(loc='upper left', fontsize=9)
        title(ax, f'{"ABC"[axs.tolist().index(ax)]}   NAND → {lab}, $N_P = 10$')
    save(fig, 'figS3_other_targets.png', outdir)


def small_pop(outdir, big, exp, exp_big, color, name, letter_titles, paper_notes):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axs[0, 0]; loglog(ax, ylabel='total mutations to AND')
    r = series(exp, 'M')
    dots(ax, r, color, label='$N_P = 10$ (mean, 5–95%)')
    fit_lines(ax, r, color, big)
    h = halves(r)
    note(ax, paper_notes[0] + (f'\nhalves: $b$ = {h[0]:.2f} / {h[1]:.2f}' if h else ''))
    ax.legend(loc='upper left', fontsize=9)
    title(ax, letter_titles[0])
    ax = axs[0, 1]; loglog(ax, ylabel='fixating mutations per run')
    rfx = series(exp, 'fix')
    dots(ax, rfx, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'fix', nmax=rfx[-1, 0] if len(rfx) else None), INK,
         label='$N_P = 10^4$', hollow=True, marker='s')
    dots(ax, series(exp, 'fix_ben'), GREEN, label='$N_P = 10$, beneficial only', marker='D',
         ms=5, bars=False)
    ax.legend(loc='upper left', fontsize=9)
    title(ax, letter_titles[1])
    ax = axs[1, 0]; loglog(ax, ylabel='$P(c)$')
    r = series(exp, 'pc')
    dots(ax, r, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'pc', nmax=r[-1, 0] if len(r) else None), INK,
         label='$N_P = 10^4$', hollow=True, marker='s', bars=False)
    fit_lines(ax, r, color, big)
    h = halves(r)
    note(ax, paper_notes[1] + (f'\nhalves: $b$ = {h[0]:.2f} / {h[1]:.2f}' if h else ''),
         loc=(0.02, 0.04), ha='left')
    ax.legend(loc='upper right', fontsize=9)
    title(ax, letter_titles[2])
    ax = axs[1, 1]; ax.set_xscale('log'); ax.set_xlabel('network size $N$')
    ax.set_ylabel('$P(b\\,|\\,c) = \\langle f_b \\rangle$'); ax.grid(True, color=GRID_C, lw=0.6)
    r = series(exp, 'fb')
    dots(ax, r, color, label='$N_P = 10$')
    dots(ax, series(exp_big, 'fb', nmax=r[-1, 0] if len(r) else None), INK,
         label='$N_P = 10^4$', hollow=True, marker='s', bars=False)
    f = fit(r)
    note(ax, paper_notes[2] + (f'\nfit exponent here: {f[1]:+.3f}' if f else ''),
         loc=(0.98, 0.95), va='top')
    ax.set_ylim(0, 0.75)
    ax.legend(loc='lower left', fontsize=9)
    title(ax, letter_titles[3])
    save(fig, name, outdir)


def figS4(outdir, big):
    small_pop(outdir, big, 'par_10', 'par_1e4', ORANGE, 'figS4_small_population.png',
              ['A   $N_P = 10$: total mutations', 'B   fixating mutations',
               'C   $P(c)$ falls, then levels off', 'D   $P(b|c)$ nearly flat'],
              ['paper: exponent 0.90 (0.87 / 0.92)', 'paper: $-0.85$ ($-0.81$ / $-0.87$)',
               'paper: $-0.020$ ($-0.049$ / $-0.012$)'])


def figS5(outdir, big):
    small_pop(outdir, big, 'chain_10', 'chain_1e4', GREEN, 'figS5_small_population_chain.png',
              ['A   chain start, $N_P = 10$: total mutations', 'B   fixating mutations',
               'C   $P(c) \\approx 1$', 'D   $P(b|c)$'],
              ['paper: $41.3\\,N^{0.10}$ (0.21 / $-0.17$)', 'paper: exponent 0.027',
               'paper: $-0.135$ ($-0.238$ / $-0.036$)'])


def figS6(outdir, big):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, exp, c, lab, exp_slope in [(axs[0], 'par_1e4', BLUE, 'A   parallel start: time is flat', 0),
                                       (axs[1], 'chain_1e4', VERM, 'B   chain start: time falls as $1/N$', -1)]:
        loglog(ax, ylabel='time to evolve AND (model time units)')
        r = series(exp, 'T')
        dots(ax, r, c, label='simulation (mean, 5–95%)')
        fit_lines(ax, r, c, big)
        ax.legend(loc='upper right' if exp_slope else 'upper left', fontsize=9)
        note(ax, '$T \\mid M \\sim$ Gamma$(M,\\ \\tau / (N_P (2N+1)))$, $\\tau = 1000$\n'
                 '$\\langle T \\rangle \\propto \\langle M \\rangle / N$', loc=(0.02, 0.04), ha='left')
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
