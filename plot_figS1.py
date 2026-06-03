"""Plot Fig S1: mutations to reach AND vs network size N."""
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

OUT_DIR = 'figS1_data'


def load(name):
    with open(os.path.join(OUT_DIR, name + '.pickle'), 'rb') as f:
        return pickle.load(f)


sizes   = np.array(load('sizes'))
means   = np.array(load('means_mutations'))
medians = np.array(load('medians_mutations'))
pct5    = np.array(load('pct5_mutations'))
pct95   = np.array(load('pct95_mutations'))

# Linear fit on log-log starting from N=10 (small N is flat/noisy)
fit_mask = sizes >= 10
log_n = np.log10(sizes[fit_mask])
log_m = np.log10(medians[fit_mask])
slope, intercept, r, *_ = linregress(log_n, log_m)
fit_x = np.array([sizes[fit_mask].min(), sizes.max()])
fit_y = 10 ** (intercept + slope * np.log10(fit_x))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ── left: log-log ────────────────────────────────────────────────────────────
ax = axes[0]
ax.fill_between(sizes, pct5, pct95, alpha=0.2, color='steelblue', label='5th–95th pct')
ax.plot(sizes, medians, 'o-', color='steelblue', lw=1.5, ms=4, label='Median')
ax.plot(sizes, means,   's--', color='tomato',    lw=1,   ms=4, label='Mean')
ax.plot(fit_x, fit_y,   'k:',  lw=1.5,
        label=f'Power-law fit (median)\n(slope={slope:.2f}, R²={r**2:.3f})')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Network size N', fontsize=12)
ax.set_ylabel('Mutations to reach AND', fontsize=12)
ax.set_title('Log–log scaling', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# ── right: linear ────────────────────────────────────────────────────────────
ax = axes[1]
ax.fill_between(sizes, pct5, pct95, alpha=0.2, color='steelblue', label='5th–95th pct')
ax.plot(sizes, medians, 'o-', color='steelblue', lw=1.5, ms=4, label='Median')
ax.plot(sizes, means,   's--', color='tomato',    lw=1,   ms=4, label='Mean')
ax.set_xlabel('Network size N', fontsize=12)
ax.set_ylabel('Mutations to reach AND', fontsize=12)
ax.set_title('Linear scale', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('Fig S1 — Random walk to AND (500 trials per N)', fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig('figS1.png', dpi=150, bbox_inches='tight')
print(f"Saved figS1.png  (N={sizes.tolist()})")
print(f"Power-law exponent: {slope:.3f}  (R²={r**2:.4f})")
plt.show()
