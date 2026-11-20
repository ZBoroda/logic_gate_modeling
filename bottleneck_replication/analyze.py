"""
Compare both engines with each other and with the bottleneck note's exact results.

Writes bottleneck_replication/fig_endpoints.png, fig_interpolating.png and
tables.md (all tables quoted in RESULTS.md).

Usage:  python -m bottleneck_replication.analyze
"""
import os
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from bottleneck_replication.run_sims import (path, configs, genome_for, NP,
                                             N_PARALLEL, N_MID, N_CHAIN, ALPHAS_MID)
from bottleneck_replication.starts import module_length, bottleneck_state
from bottleneck_replication.theory import exact_M_parallel, inv_p0_formula, enumerate_p

HERE = os.path.dirname(__file__)
ENGINES = ('faster', 'fast')
LABEL = {'faster': 'faster_circuit', 'fast': 'fast_circuit'}
# categorical slots 1..8 of the reference palette, fixed order
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ENGINE_COLOR = {'faster': SLOTS[0], 'fast': SLOTS[1]}
ALPHA_COLOR = {a: SLOTS[i] for i, a in enumerate([0.0] + ALPHAS_MID + [1.0])}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load(engine, alpha, N):
    p = path(engine, alpha, N)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {'M': d['M'], 'first': d['counts'][:, 0], 'seconds': float(d['seconds'])}


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))


def fit_exponent(Ns, means, sems):
    """Weighted least squares of ln<M> on ln N; sigma(ln<M>) = SEM / mean."""
    x = np.log(Ns); y = np.log(means); s = np.asarray(sems) / np.asarray(means)
    w = 1 / s ** 2
    W = w.sum(); xb = (w * x).sum() / W; yb = (w * y).sum() / W
    Sxx = (w * (x - xb) ** 2).sum()
    slope = (w * (x - xb) * (y - yb)).sum() / Sxx
    return slope, math.sqrt(1 / Sxx)


def series(engine, alpha, Ns, key='M'):
    out = []
    for N in Ns:
        d = load(engine, alpha, N)
        if d is not None:
            m, s = mean_sem(d[key])
            out.append((N, m, s, float(np.median(d[key])), len(d[key])))
    return out


def style(ax):
    ax.grid(True, which='major', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def errplot(ax, data, color, filled, label, ms=6):
    if not data:
        return
    N, m, s = (np.array([r[i] for r in data]) for i in range(3))
    ax.errorbar(N, m, yerr=s, fmt='o' if filled else 's', ms=ms, color=color,
                mfc=color if filled else 'white', mec=color, mew=1.5, lw=1.2,
                capsize=0, label=label, zorder=3)


# ── tables ────────────────────────────────────────────────────────────────────

def z(a, b):
    return (a[1] - b[1]) / math.hypot(a[2], b[2])


def table_parallel():
    rows = ['| N | exact ⟨M⟩ (Eq. 3) | faster_circuit | z vs exact | fast_circuit | z vs exact | z (fast − faster) |',
            '|---|---|---|---|---|---|---|']
    zs = {'faster': [], 'fast': [], 'diff': []}
    for N in N_PARALLEL:
        ex = exact_M_parallel(N, NP)
        r = {e: series(e, 0.0, [N]) for e in ENGINES}
        cells = [f'{N}', f'{ex:.1f}']
        for e in ENGINES:
            if r[e]:
                _, m, s, _, _ = r[e][0]
                zz = (m - ex) / s
                zs[e].append(zz)
                cells += [f'{m:.1f} ± {s:.1f}', f'{zz:+.2f}']
            else:
                cells += ['–', '–']
        if r['faster'] and r['fast']:
            dz = z(r['fast'][0], r['faster'][0]); zs['diff'].append(dz)
            cells.append(f'{dz:+.2f}')
        else:
            cells.append('–')
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows), zs


def table_chain():
    rows = ['| N | faster_circuit mean ± SEM | median | fast_circuit mean ± SEM | median | z (fast − faster) |',
            '|---|---|---|---|---|---|']
    zs = []
    for N in N_CHAIN:
        r = {e: series(e, 1.0, [N]) for e in ENGINES}
        cells = [f'{N}']
        for e in ENGINES:
            cells += ([f'{r[e][0][1]:.1f} ± {r[e][0][2]:.1f}', f'{r[e][0][3]:.0f}'] if r[e] else ['–', '–'])
        if r['faster'] and r['fast']:
            dz = z(r['fast'][0], r['faster'][0]); zs.append(dz); cells.append(f'{dz:+.2f}')
        else:
            cells.append('–')
        rows.append('| ' + ' | '.join(cells) + ' |')
    pooled = {e: np.concatenate([load(e, 1.0, N)['M'] for N in N_CHAIN if load(e, 1.0, N) is not None] or [np.zeros(1)])
              for e in ENGINES}
    extra = '\n\nPooled over N: ' + '; '.join(
        f'{LABEL[e]} mean {pooled[e].mean():.1f}, median {np.median(pooled[e]):.0f}, '
        f'P(M > 1000) = {100 * (pooled[e] > 1000).mean():.2f}%' for e in ENGINES)
    return '\n'.join(rows) + extra, zs


def table_first_step():
    rows = ['| α | N | L | exact 1/p₀ (Eq. 4) | faster_circuit | z | fast_circuit | z | z (fast − faster) |',
            '|---|---|---|---|---|---|---|---|---|']
    zs = {'faster': [], 'fast': [], 'diff': []}
    for a in ALPHAS_MID:
        for N in N_MID:
            L = module_length(N, a)
            ex = inv_p0_formula(N, L, NP)
            r = {e: series(e, a, [N], 'first') for e in ENGINES}
            cells = [f'{a}', f'{N}', f'{L}', f'{ex:.1f}']
            for e in ENGINES:
                if r[e]:
                    _, m, s, _, _ = r[e][0]; zz = (m - ex) / s; zs[e].append(zz)
                    cells += [f'{m:.1f} ± {s:.1f}', f'{zz:+.2f}']
                else:
                    cells += ['–', '–']
            if r['faster'] and r['fast']:
                dz = z(r['fast'][0], r['faster'][0]); zs['diff'].append(dz); cells.append(f'{dz:+.2f}')
            else:
                cells.append('–')
            rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows), zs


def table_total():
    rows = ['| α | N | faster_circuit ⟨M⟩ | fast_circuit ⟨M⟩ | z (fast − faster) |', '|---|---|---|---|---|']
    zs = []
    for a in ALPHAS_MID:
        for N in N_MID:
            r = {e: series(e, a, [N]) for e in ENGINES}
            cells = [f'{a}', f'{N}']
            for e in ENGINES:
                cells.append(f'{r[e][0][1]:.0f} ± {r[e][0][2]:.0f}' if r[e] else '–')
            if r['faster'] and r['fast']:
                dz = z(r['fast'][0], r['faster'][0]); zs.append(dz); cells.append(f'{dz:+.2f}')
            else:
                cells.append('–')
            rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows), zs


NOTE_TABLE3 = {0.0: (None, 0.99), 0.2: (0.92, 0.97), 0.4: (0.69, 0.90), 0.5: (0.53, 0.89),
               0.6: (0.50, 0.81), 0.75: (0.30, 0.57), 0.9: (0.11, 0.19), 1.0: (None, 0.04)}


def exponents(engine, a, lo, hi, key):
    Ns = {0.0: N_PARALLEL, 1.0: N_CHAIN}.get(a, N_MID)
    data = [r for r in series(engine, a, [N for N in Ns if lo <= N <= hi], key)]
    if len(data) < 3:
        return None
    return fit_exponent([r[0] for r in data], [r[1] for r in data], [r[2] for r in data])


def note_range(a):
    return {0.0: (100, 1000), 1.0: (100, 501)}.get(a, (100, 800))


def table_exponents():
    rows = ['| α | β(α) | note: total | faster: total (note range) | fast: total (note range) '
            '| faster: total (100–10⁴) | fast: total (100–10⁴) | 1−α | note: first step '
            '| faster: first step (note range) | fast: first step (note range) |',
            '|' + '---|' * 11]
    fmt = lambda r: f'{r[0]:.2f} ± {r[1]:.2f}' if r else '–'
    out = {}
    for a in [0.0] + ALPHAS_MID + [1.0]:
        lo, hi = note_range(a)
        beta = min(1, 2 * (1 - a))
        tot = {e: exponents(e, a, lo, hi, 'M') for e in ENGINES}
        ext = {e: exponents(e, a, 100, 10_001, 'M') for e in ENGINES}
        fs = {e: (exponents(e, a, lo, hi, 'first') if 0 < a < 1 else None) for e in ENGINES}
        out[a] = (tot, ext, fs)
        nf, nt = NOTE_TABLE3[a]
        rows.append(f'| {a} | {beta:.2f} | {nt:.2f} | {fmt(tot["faster"])} | {fmt(tot["fast"])} '
                    f'| {fmt(ext["faster"])} | {fmt(ext["fast"])} | {1 - a:.2f} '
                    f'| {nf if nf is not None else "–"} | {fmt(fs["faster"])} | {fmt(fs["fast"])} |')
    return '\n'.join(rows), out


def table_bottleneck():
    note = {(0.5, 100): 416, (0.5, 800): 2652, (0.6, 100): 348, (0.6, 800): 1861,
            (0.75, 100): 192, (0.75, 800): 626}
    rows = ['| α | N | L | 1/p₀ | 1/p_T (enum.) | sum | note: sim ⟨M⟩ | faster ⟨M⟩ | fast ⟨M⟩ |',
            '|---|---|---|---|---|---|---|---|---|']
    for (a, N), nm in note.items():
        L = module_length(N, a)
        k = L // 2 if (L // 2) % 2 else L // 2 + 1
        ip0 = inv_p0_formula(N, L, NP)
        ipT = 1 / enumerate_p(bottleneck_state(N, L, k), NP)[0]
        cells = [f'{a}', f'{N}', f'{L}', f'{ip0:.0f}', f'{ipT:.0f}', f'{ip0 + ipT:.0f}', f'{nm}']
        for e in ENGINES:
            r = series(e, a, [N])
            cells.append(f'{r[0][1]:.0f} ± {r[0][2]:.0f}' if r else '–')
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def table_cost():
    """CPU time per mutation. fast: pool of workers, so wall x workers / sum(M)."""
    rows = ['| α | N | faster_circuit (µs/mutation) | fast_circuit (µs/mutation, ≈ wall × 10 workers) |',
            '|---|---|---|---|']
    for a, N in [(0.0, 100), (0.0, 1000), (0.0, 10000), (0.5, 10000), (1.0, 10001)]:
        c = []
        for e, k in (('faster', 1), ('fast', 10)):
            d = load(e, a, N)
            c.append(f'{d["seconds"] * k / d["M"].sum() * 1e6:.2f}' if d else '–')
        rows.append(f'| {a} | {N} | {c[0]} | {c[1]} |')
    return '\n'.join(rows)


def chi2_summary(name, zs):
    zs = np.asarray(zs)
    if len(zs) == 0:
        return f'- {name}: no data'
    chi2 = float((zs ** 2).sum())
    p = 1 - stats.chi2.cdf(chi2, len(zs))
    return (f'- {name}: {len(zs)} points, mean z = {zs.mean():+.2f}, '
            f'χ² = {chi2:.1f} (p = {p:.2f}), max |z| = {np.abs(zs).max():.2f}')


# ── figures ───────────────────────────────────────────────────────────────────

def fig_endpoints():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    # A: <M> vs N, parallel and chain
    ax = axs[0]
    Nl = np.logspace(np.log10(5), 4, 200)
    ax.plot(Nl, [exact_M_parallel(n, NP) for n in Nl], color=INK, lw=2, label='parallel: exact Eq. 3')
    for e in ENGINES:
        errplot(ax, series(e, 0.0, N_PARALLEL), ENGINE_COLOR[e], e == 'faster', f'parallel, {LABEL[e]}')
        ch = series(e, 1.0, N_CHAIN)
        errplot(ax, ch, ENGINE_COLOR[e], e == 'faster', None)
        ax.plot([r[0] for r in ch], [r[3] for r in ch], ls='none', marker='_', ms=10, mew=2,
                color=ENGINE_COLOR[e], label=f'chain median, {LABEL[e]}')
    ax.text(12, 70, 'chain (α = 1)', color=MUTED, fontsize=9)
    ax.set(xscale='log', yscale='log', xlabel='network size N', ylabel='mutations to evolve AND, ⟨M⟩',
           title='A  endpoints: parallel and chain')
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    style(ax)

    # B: parallel, simulation / exact
    ax = axs[1]
    for e in ENGINES:
        d = series(e, 0.0, N_PARALLEL)
        if d:
            N = np.array([r[0] for r in d]); m = np.array([r[1] for r in d]); s = np.array([r[2] for r in d])
            ex = np.array([exact_M_parallel(n, NP) for n in N])
            off = 1.06 if e == 'fast' else 1 / 1.06
            ax.errorbar(N * off, m / ex, yerr=s / ex, fmt='o' if e == 'faster' else 's', ms=6,
                        color=ENGINE_COLOR[e], mfc=ENGINE_COLOR[e] if e == 'faster' else 'white',
                        mew=1.5, lw=1.2, label=LABEL[e], zorder=3)
    Nl = np.logspace(np.log10(4), 4.1, 100)
    ax.plot(Nl, 1 + 1 / Nl, color=ENGINE_COLOR['fast'], lw=1.2, ls='--',
            label='1 + 1/N (counted no-ops, est.)')
    ax.axhline(1, color=INK, lw=1.5)
    ax.set(xscale='log', xlabel='network size N', ylabel='simulated ⟨M⟩ / exact ⟨M⟩',
           title='B  parallel start: ratio to exact theory')
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    # C: first step / exact for every alpha, pooled N, both engines
    ax = axs[2]
    for e in ENGINES:
        for i, a in enumerate(ALPHAS_MID):
            d = series(e, a, N_MID, 'first')
            if not d:
                continue
            N = np.array([r[0] for r in d]); m = np.array([r[1] for r in d]); s = np.array([r[2] for r in d])
            ex = np.array([inv_p0_formula(n, module_length(n, a), NP) for n in N])
            off = 1.04 if e == 'fast' else 1 / 1.04
            ax.errorbar(N * off, m / ex, yerr=s / ex, fmt='o' if e == 'faster' else 's', ms=5,
                        color=ALPHA_COLOR[a], mfc=ALPHA_COLOR[a] if e == 'faster' else 'white',
                        mew=1.3, lw=1, label=f'α = {a}' if e == 'faster' else None, zorder=3)
    ax.axhline(1, color=INK, lw=1.5)
    ax.set(xscale='log', xlabel='network size N', ylabel='simulated first step / exact 1/p₀',
           title='C  first step vs Eq. 4 (filled: faster, open: fast)')
    ax.legend(fontsize=8, frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_endpoints.png'), dpi=150)
    plt.close(fig)


def fig_interpolating(exps):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    # A: first step with exact curve at actual L(N)
    ax = axs[0]
    Nl = np.unique(np.logspace(2, 4, 400).astype(int))
    for a in ALPHAS_MID:
        ax.plot(Nl, [inv_p0_formula(n, module_length(n, a), NP) for n in Nl],
                color=ALPHA_COLOR[a], lw=1.2, label=f'α = {a}')
        for e in ENGINES:
            errplot(ax, series(e, a, N_MID, 'first'), ALPHA_COLOR[a], e == 'faster', None, ms=5)
    ax.set(xscale='log', yscale='log', xlabel='network size N', ylabel='mutations spent at f = 0',
           title='A  first step (lines: exact 1/p₀, no fit)')
    ax.legend(fontsize=8, frameon=False, ncol=2)
    style(ax)

    # B: total
    ax = axs[1]
    Nl2 = np.logspace(np.log10(5), 4, 200)
    ax.plot(Nl2, [exact_M_parallel(n, NP) for n in Nl2], color=ALPHA_COLOR[0.0], lw=1.2, label='α = 0 (exact)')
    for a in [0.0] + ALPHAS_MID + [1.0]:
        Ns = {0.0: N_PARALLEL, 1.0: N_CHAIN}.get(a, N_MID)
        for e in ENGINES:
            d = series(e, a, Ns)
            errplot(ax, d, ALPHA_COLOR[a], e == 'faster', None, ms=5)
        d = series('faster', a, Ns)
        if d and 0 < a <= 1:
            beta = min(1, 2 * (1 - a))
            N0, m0 = d[0][0], d[0][1]
            ax.plot([N0, 1e4], [m0, m0 * (1e4 / N0) ** beta], color=ALPHA_COLOR[a], lw=1, ls='--',
                    label=f'α = {a}, slope β = {beta:.2g}')
    ax.set(xscale='log', yscale='log', xlabel='network size N', ylabel='total mutations to AND, ⟨M⟩',
           title='B  total (dashed: slope min{1, 2(1−α)})')
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
    style(ax)

    # C: exponents
    ax = axs[2]
    al = np.linspace(0, 1, 101)
    ax.plot(al, 1 - al, color=MUTED, lw=1.2, label='first step: 1 − α')
    ax.plot(al, np.minimum(1, 2 * (1 - al)), color=INK, lw=1.2, ls='--', label='total: β(α)')
    for e, dx in (('faster', -0.012), ('fast', 0.012)):
        c = ENGINE_COLOR[e]
        for a, (tot, ext, fs) in exps.items():
            for r, mk, fill, lab in ((tot[e], 'o', True, f'total, note range ({LABEL[e]})'),
                                     (ext[e], 'D', False, f'total, 100–10⁴ ({LABEL[e]})'),
                                     (fs[e], 's', False, f'first step ({LABEL[e]})')):
                if r is None:
                    continue
                ax.errorbar(a + dx, r[0], yerr=r[1], fmt=mk, ms=6, color=c, mfc=c if fill else 'white',
                            mew=1.3, lw=1, label=lab if a == 0.5 else None, zorder=3)
    nx = [a for a in NOTE_TABLE3]
    ax.plot(nx, [NOTE_TABLE3[a][1] for a in nx], ls='none', marker='x', ms=7, color=INK, mew=1.5,
            label='note: total (Table 3)')
    ax.set(xlabel='module exponent α  (L ≃ N^α)', ylabel='fitted scaling exponent',
           title='C  exponents', ylim=(-0.25, 1.15))
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_interpolating.png'), dpi=150)
    plt.close(fig)


def main():
    t_par, z_par = table_parallel()
    t_ch, z_ch = table_chain()
    t_fs, z_fs = table_first_step()
    t_tot, z_tot = table_total()
    t_exp, exps = table_exponents()
    parts = [
        '## Parallel start: ⟨M⟩ vs exact Eq. 3\n\n' + t_par,
        '## Chain start (α = 1)\n\n' + t_ch,
        '## First step: mutations at f = 0 vs exact 1/p₀ (Eq. 4)\n\n' + t_fs,
        '## Interpolating family: total ⟨M⟩\n\n' + t_tot,
        '## Fitted exponents\n\n' + t_exp,
        '## Two-step estimate vs simulation (note Table 4)\n\n' + table_bottleneck(),
        '## Cost per mutation (from run wall times)\n\n' + table_cost(),
        '## Agreement summary (z-scores)\n\n' + '\n'.join([
            chi2_summary('parallel, faster_circuit vs Eq. 3', z_par['faster']),
            chi2_summary('parallel, fast_circuit vs Eq. 3', z_par['fast']),
            chi2_summary('first step, faster_circuit vs Eq. 4', z_fs['faster']),
            chi2_summary('first step, fast_circuit vs Eq. 4', z_fs['fast']),
            chi2_summary('engines, parallel ⟨M⟩', z_par['diff']),
            chi2_summary('engines, chain ⟨M⟩', z_ch),
            chi2_summary('engines, first step', z_fs['diff']),
            chi2_summary('engines, interpolating total ⟨M⟩', z_tot),
        ]),
    ]
    with open(os.path.join(HERE, 'tables.md'), 'w') as f:
        f.write('# Replication tables (generated by analyze.py)\n\n' + '\n\n'.join(parts) + '\n')
    fig_endpoints()
    fig_interpolating(exps)
    print('\n\n'.join(parts))


if __name__ == '__main__':
    main()
