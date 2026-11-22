"""Fig S1B, why per-proposal P(c) is about half the computing-edge fraction in the random walk.
Usage: python -m paper_replication.rw_cycle N steps   (e.g. 1000 20000)"""
import sys, random
import numpy as np
from bottleneck_replication.starts import parallel
from faster_circuit.observe import _module
N = int(sys.argv[1]); steps = int(sys.argv[2]); random.seed(1)
g = list(np.asarray(parallel(N)))
E = len(g); nodes = N + 2
def anc_of(src, g):          # gates reachable upstream from node src
    seen, st = set(), [src]
    while st:
        v = st.pop()
        if v < 2 or v in seen: continue
        seen.add(v); st += [g[2*(v-2)], g[2*(v-2)+1]]
    return seen
def cyclic(g, e, s):
    if e == E - 1: return False
    gate = e // 2 + 2
    return gate in anc_of(s, g) or s == gate
def propose(g):
    while True:
        e = random.randrange(E); gate = None if e == E-1 else e//2+2
        s = random.choice([v for v in range(nodes) if v != g[e] and v != gate])
        if not cyclic(g, e, s): return e, s
for _ in range(steps):
    e, s = propose(g); g[e] = s
_, mod = _module(g); mod = set(int(x) for x in mod)
comp = [e for e in range(E) if e == E-1 or (e//2+2) in mod]
ok = {True: [0, 0], False: [0, 0]}
for _ in range(20000):
    e = random.randrange(E); gate = None if e == E-1 else e//2+2
    s = random.choice([v for v in range(nodes) if v != g[e] and v != gate])
    c = e in set(comp); ok[c][0] += 1; ok[c][1] += not cyclic(g, e, s)
fc = len(comp)/E
ac, an = ok[True][1]/ok[True][0], ok[False][1]/ok[False][0]
print(f'N={N}: computing edge fraction {fc:.3f}; acyclic share: computing {ac:.3f}, other {an:.3f}; '
      f'predicted per-proposal P(c) {fc*ac/(fc*ac+(1-fc)*an):.3f}')
