"""
Intermediate-exponents figure (bottleneck note, Fig. 2) extended to N = 10^6
with faster_circuit. Reads results_1e6/, writes fig_intermediate_exponents_1e6.png
and prints the fitted exponents.

Usage:  python -m bottleneck_replication.fig_1e6
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bottleneck_replication.run_1e6 import (path, ALPHAS_MID, N_GRID, N_PARALLEL, N_CHAIN, NP)
from bottleneck_replication.starts import module_length
from bottleneck_replication.theory import (exact_M_parallel, inv_p0_formula,
                                           p_chain_reservoir, bottleneck_k)

HERE = os.path.dirname(__file__)
COL = {0.0: '#0072B2', 0.2: '#2E2A7A', 0.4: '#8F9A3A', 0.5: '#CC79A7', 0.6: '#E69F00',
       0.75: '#009E73', 0.9: '#56B4E9', 1.0: '#D55E00'}
INK, MUTED = '#1a1a1a', '#6b6b6b'
FIT_ALL = (100, 1_000_001)          # total and first-step fits
FIT_HI = (10_000, 1_000_001)        # "asymptotic" fits


def load(alpha, N):
    p = path(alpha, N)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d['M'].astype(float), d['counts'][:, 0].astype(float)


def stats(alpha, Ns, which):
    rows = []
    for N in Ns:
        d = load(alpha, N)
        if d is None:
            continue
        x = d[0] if which == 'M' else d[1]
        rows.append((N, x.mean(), x.std(ddof=1) / math.sqrt(len(x)), np.median(x)))
    return np.array(rows) if rows else np.zeros((0, 4))


def wfit(N, y, s=None):
    """Weighted LS slope of ln y on ln N, with its standard error."""
    x = np.log(N); ly = np.log(y)
    w = np.ones_like(x) if s is None else (np.asarray(y) / np.asarray(s)) ** 2
    W = w.sum(); xb = (w * x).sum() / W; yb = (w * ly).sum() / W
    Sxx = (w * (x - xb) ** 2).sum()
    b = (w * (x - xb) * (ly - yb)).sum() / Sxx
    if s is None:
        res = ly - yb - b * (x - xb)
        return b, math.sqrt((res ** 2).sum() / max(len(x) - 2, 1) / Sxx)
    return b, math.sqrt(1 / Sxx)


def in_range(rows, lo, hi):
    return rows[(rows[:, 0] >= lo) & (rows[:, 0] <= hi)] if len(rows) else rows


def two_step(N, a):
    L = module_length(N, a)
    return inv_p0_formula(N, L, NP) + 1 / p_chain_reservoir(N, L, bottleneck_k(L), NP)[0]


def style(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=13)
    ax.tick_params(labelsize=10)


# ── panel A: schematic ────────────────────────────────────────────────────────

def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    style(ax, 'A   starting genotype: chain module in a reservoir')
    orange, grey = COL[1.0], '#a0a0a0'

    def box(x, y, label, fill='white', w=0.75, h=0.5):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='square,pad=0',
                                    fc=fill, ec=INK, lw=1.2, zorder=3))
        ax.text(x, y, label, ha='center', va='center', fontsize=10, zorder=4, style='italic'
                if label in ('L', 'O') else 'normal')

    I1, I2 = (2.4, 8.9), (4.6, 8.9)
    box(*I1, '$I_1$', '#d9d9d9'); box(*I2, '$I_2$', '#d9d9d9')
    chain_y = [7.8, 7.0, 6.2]
    for i, y in enumerate(chain_y):
        box(1.8, y, str(i + 1))
    box(1.8, 4.6, 'L'); box(1.8, 3.6, 'O', '#d9d9d9')
    ax.plot([I1[0] - 0.2, 1.9], [I1[1] - 0.25, chain_y[0] + 0.25], color=orange, lw=1.8)
    ax.plot([I2[0] - 0.3, 2.1], [I2[1] - 0.25, chain_y[0]], color=orange, lw=1.8)
    for y0, y1 in zip(chain_y[:-1], chain_y[1:]):
        for dx in (-0.06, 0.06):
            ax.plot([1.8 + dx] * 2, [y0 - 0.25, y1 + 0.25], color=orange, lw=1.6)
    for dx in (-0.06, 0.06):
        ax.plot([1.8 + dx] * 2, [chain_y[-1] - 0.25, 4.85], color=orange, lw=1.6, ls=':')
    ax.plot([1.8, 1.8], [4.35, 3.85], color=orange, lw=1.6)
    res_x = [5.0, 6.2, 8.4]
    for x in res_x:
        box(x, 7.0, '')
        for src in (I1, I2):
            ax.plot([src[0] + 0.3, x], [src[1] - 0.25, 7.25], color=grey, lw=1, zorder=1)
    ax.text(7.3, 7.0, '...', ha='center', va='center', fontsize=12)
    ax.annotate('', xy=(0.75, 4.4), xytext=(0.75, 8.1),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2))
    ax.text(0.45, 6.25, r'$L \simeq N^{\alpha}$ (odd)', rotation=90, ha='center', va='center', fontsize=10)
    ax.text(9.6, 5.9, r'$N - L$ pristine copies of NAND$(I_1, I_2)$', ha='right', fontsize=10)
    for i, t in enumerate(['gate 1 = NAND, gates 2...L = NOT',
                           'output = NAND (fitness 0 for AND)',
                           r'$\alpha = 0$: parallel network (Fig. 3A)',
                           r'$\alpha = 1$: chain (Fig. 5A)']):
        ax.text(9.6, 4.8 - 0.75 * i, t, ha='right', fontsize=9.5, color='#404040')


# ── figure ────────────────────────────────────────────────────────────────────

def main():
    fig, axs = plt.subplots(2, 2, figsize=(14.5, 11.2))
    panel_a(axs[0, 0])

    # B: first step
    ax = axs[0, 1]
    Nl = np.unique(np.round(np.logspace(np.log10(20), 6, 1500)).astype(int))
    for a in ALPHAS_MID:
        ax.plot(Nl, [inv_p0_formula(n, module_length(n, a), NP) for n in Nl], color=COL[a], lw=1.4)
        r = stats(a, N_GRID, 'first')
        if len(r):
            ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt='o', ms=6, color=COL[a],
                        mec=INK, mew=0.6, lw=1, zorder=3)
        ax.text(1.35e6, inv_p0_formula(10**6, module_length(10**6, a), NP), f'$\\alpha={a}$',
                color=COL[a], va='center', fontsize=10)
    ax.plot([], [], color='#444', lw=1.4, label='exact $1/p_0$ at actual $L(N)$ (no fit)')
    ax.plot([], [], 'o', color='#999', mec=INK, mew=0.6, label='simulation, mean ± SEM')
    ax.set(xscale='log', yscale='log', xlim=(15, 4e6), xlabel='network size $N$',
           ylabel='mutations spent at $f = 0$')
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    style(ax, 'B   first step: exact local geometry')

    # C: totals
    ax = axs[1, 0]
    Nl2 = np.logspace(np.log10(4), 6.1, 300)
    ax.plot(Nl2, [exact_M_parallel(n, NP) for n in Nl2], color=COL[0.0], lw=1.8)
    r = stats(0.0, N_PARALLEL, 'M')
    if len(r):
        ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt='o', ms=6, color=COL[0.0], mec=INK,
                    mew=0.6, lw=1, zorder=3, label=r'$\alpha=0$ (+ exact curve)')
    for a in [0.5, 0.6, 0.75, 0.9]:
        r = stats(a, N_GRID, 'M')
        if not len(r):
            continue
        ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt='o', ms=6, color=COL[a], mec=INK,
                    mew=0.6, lw=1, zorder=3, label=f'$\\alpha={a}$')
        beta = min(1, 2 * (1 - a))
        hi = r[-1]
        Ng = np.array([25, hi[0]])
        ax.plot(Ng, hi[1] * (Ng / hi[0]) ** beta, color=COL[a], lw=1.2, ls='--')
    r = stats(1.0, N_CHAIN, 'M')
    if len(r):
        ax.errorbar(r[:, 0], r[:, 1], yerr=r[:, 2], fmt='s', ms=6, color=COL[1.0], mec=INK,
                    mew=0.6, lw=1, zorder=3, label=r'$\alpha=1$ mean')
        ax.plot(r[:, 0], r[:, 3], 's', ms=6, mfc='white', mec=COL[1.0], mew=1.3,
                label=r'$\alpha=1$ median')
    ax.set(xscale='log', yscale='log', xlabel='network size $N$',
           ylabel=r'total mutations to AND, $\langle M \rangle$')
    ax.legend(frameon=False, fontsize=10, ncol=2, loc='upper left')
    ax.text(0.98, 0.03, r'dashed: slope $\min\{1, 2(1-\alpha)\}$', transform=ax.transAxes,
            ha='right', fontsize=9, color=MUTED)
    style(ax, 'C   total: set by the bottleneck step')

    # D: exponents
    ax = axs[1, 1]
    al = np.linspace(0, 1, 101)
    ax.plot(al, 1 - al, color='#777', lw=1.4, label=r'$1-\alpha$ (first step, asymptotic)')
    ax.plot(al, np.minimum(1, 2 * (1 - al)), color=INK, lw=1.4, ls='--',
            label=r'$\beta(\alpha)$ (total, asymptotic)')
    table = []
    for a in [0.0] + ALPHAS_MID + [1.0]:
        Ns = {0.0: N_PARALLEL, 1.0: N_CHAIN}.get(a, N_GRID)
        c = COL[a]
        tot = stats(a, Ns, 'M')
        row = {'alpha': a}
        t_all = in_range(tot, *FIT_ALL)
        if len(t_all) >= 3:
            row['total'] = wfit(t_all[:, 0], t_all[:, 1], t_all[:, 2])
            ax.errorbar(a - 0.012, *row['total'], fmt='o', ms=8, color=c, mec=INK, mew=0.6, zorder=4)
        t_hi = in_range(tot, *FIT_HI)
        if len(t_hi) >= 3:
            row['total_hi'] = wfit(t_hi[:, 0], t_hi[:, 1], t_hi[:, 2])
            ax.errorbar(a + 0.012, *row['total_hi'], fmt='D', ms=7, mfc='white', mec=c,
                        mew=1.6, color=c, zorder=4)
        if 0 < a < 1:
            fs = in_range(stats(a, N_GRID, 'first'), *FIT_ALL)
            if len(fs) >= 3:
                row['first'] = wfit(fs[:, 0], fs[:, 1], fs[:, 2])
                ax.errorbar(a - 0.012, *row['first'], fmt='s', ms=8, mfc='white', mec=c,
                            mew=1.6, color=c, zorder=4)
            Ns_fit = [n for n in N_GRID if FIT_ALL[0] <= n <= FIT_ALL[1]]
            row['first_exact'] = wfit(np.array(Ns_fit),
                                      np.array([inv_p0_formula(n, module_length(n, a), NP) for n in Ns_fit]))
            ax.plot(a + 0.012, row['first_exact'][0], 'x', ms=8, mew=1.8, color='#333', zorder=5)
            Nh = [n for n in N_GRID if FIT_HI[0] <= n <= FIT_HI[1]]
            row['two_step_hi'] = wfit(np.array(Nh), np.array([two_step(n, a) for n in Nh]))
            ax.plot(a + 0.03, row['two_step_hi'][0], '+', ms=11, mew=2, color='#333', zorder=5)
        table.append(row)
    for mk, kw, lab in [('s', dict(mfc='white', mec='#555', mew=1.6), 'first step, simulation'),
                        ('x', dict(color='#333', mew=1.8), 'first step, exact $1/p_0$'),
                        ('o', dict(color='#888', mec=INK, mew=0.6), r'total, simulation ($10^2$–$10^6$)'),
                        ('D', dict(mfc='white', mec='#555', mew=1.6), r'total, simulation ($10^4$–$10^6$)'),
                        ('+', dict(color='#333', mew=2, ms=11), r'total, two-step theory ($10^4$–$10^6$)')]:
        ax.plot([], [], mk, ls='none', ms=kw.pop('ms', 8), label=lab, **kw)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.08, 1.12), xlabel=r'module exponent $\alpha$  ($L \simeq N^\alpha$)',
           ylabel='scaling exponent')
    ax.legend(frameon=False, fontsize=9.5, loc='lower left')
    style(ax, r'D   exponents ($10^2 \leq N \leq 10^6$)')

    fig.tight_layout()
    out = os.path.join(HERE, 'fig_intermediate_exponents_1e6.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)

    fmt = lambda r: f'{r[0]:.3f} ± {r[1]:.3f}' if r else '–'
    print('\n| α | 1−α | first step sim | first step exact 1/p₀ | β(α) | total (10²–10⁶) | total (10⁴–10⁶) | two-step theory (10⁴–10⁶) |')
    print('|---|---|---|---|---|---|---|---|')
    for row in table:
        a = row['alpha']
        print(f"| {a} | {1 - a:.2f} | {fmt(row.get('first'))} | {fmt(row.get('first_exact'))} "
              f"| {min(1, 2 * (1 - a)):.2f} | {fmt(row.get('total'))} | {fmt(row.get('total_hi'))} "
              f"| {fmt(row.get('two_step_hi'))} |")


if __name__ == '__main__':
    main()
