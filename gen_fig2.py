"""
Generate Fig 2: genotype graph for size-2 NAND networks.

Nodes  = all canonical acyclic genotypes (gate inputs stored as unordered pairs).
Edges  = single point mutation (one input wiring or the output gate changes).
Color  = fitness for AND target (viridis: purple=0 → yellow=1).
"""
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from fast_circuit.circuit import (
    _topo_order_numba, _computing_order_numba, _fitness_numba,
    _target_array, warmup, NUM_INPUTS,
)
from goals import and_funct

SIZE = 2
N_NODES = NUM_INPUTS + SIZE  # 4 total nodes (2 inputs + 2 gates)


# ── helpers ───────────────────────────────────────────────────────────────────

def _genome(g0, g1, out):
    return np.array([g0[0], g0[1], g1[0], g1[1], out], dtype=np.int32)


def is_acyclic(g0, g1, out):
    _, fl = _topo_order_numba(_genome(g0, g1, out), NUM_INPUTS)
    return fl == N_NODES


def compute_fitness(g0, g1, out, target):
    genome = _genome(g0, g1, out)
    fo, fl = _topo_order_numba(genome, NUM_INPUTS)
    if fl < N_NODES:
        return 0.0
    eo, el = _computing_order_numba(genome, NUM_INPUTS, fo, fl)
    return float(_fitness_numba(genome, NUM_INPUTS, eo, el, target))


def shares_one(pair_a, pair_b):
    """True when two sorted input pairs differ by exactly one element (1 mutation)."""
    remaining = list(pair_b)
    count = 0
    for x in pair_a:
        if x in remaining:
            count += 1
            remaining.remove(x)
    return count == 1


# ── enumerate valid genotypes ─────────────────────────────────────────────────

def enumerate_genotypes():
    # Unordered input pairs for each gate, excluding the gate's own index (self-loop)
    g0_pairs = sorted(
        (a, b) for a in range(N_NODES) for b in range(a, N_NODES)
        if a != 2 and b != 2   # gate 0 is node 2
    )
    g1_pairs = sorted(
        (a, b) for a in range(N_NODES) for b in range(a, N_NODES)
        if a != 3 and b != 3   # gate 1 is node 3
    )
    outputs = list(range(NUM_INPUTS, NUM_INPUTS + SIZE))  # [2, 3]

    valid = [
        (g0, g1, out)
        for g0, g1, out in itertools.product(g0_pairs, g1_pairs, outputs)
        if is_acyclic(g0, g1, out)
    ]
    return valid, g0_pairs, g1_pairs, outputs


# ── build mutation graph ──────────────────────────────────────────────────────

def build_graph(genotypes, g0_pairs, g1_pairs, outputs):
    valid_set = set(genotypes)
    G = nx.Graph()
    G.add_nodes_from(genotypes)

    for g0, g1, out in genotypes:
        # Mutate gate 0's input pair (change one wiring)
        for new_g0 in g0_pairs:
            if shares_one(g0, new_g0):
                nb = (new_g0, g1, out)
                if nb in valid_set:
                    G.add_edge((g0, g1, out), nb)

        # Mutate gate 1's input pair
        for new_g1 in g1_pairs:
            if shares_one(g1, new_g1):
                nb = (g0, new_g1, out)
                if nb in valid_set:
                    G.add_edge((g0, g1, out), nb)

        # Mutate output gate
        for new_out in outputs:
            if new_out != out:
                nb = (g0, g1, new_out)
                if nb in valid_set:
                    G.add_edge((g0, g1, out), nb)

    return G


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(G, fitness, out_path='fig2.png'):
    cmap = plt.get_cmap('viridis')
    fitness_cats = [0.0, 0.25, 0.5, 0.75, 1.0]

    node_colors = [cmap(fitness[g]) for g in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 9))

    pos = nx.spring_layout(G, k=1.8, iterations=200)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#555555',
                           width=0.9, alpha=0.55, ax=ax)

    patches = [
        mpatches.Patch(facecolor=cmap(f), label=str(f))
        for f in fitness_cats
    ]
    ax.legend(handles=patches, title='Fitness Categories',
              loc='upper right', fontsize=12, title_fontsize=13,
              frameon=True)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Warming up Numba JIT...")
    warmup()

    target = _target_array(and_funct)

    genotypes, g0_pairs, g1_pairs, outputs = enumerate_genotypes()
    print(f"Valid canonical genotypes: {len(genotypes)}")

    fitness = {g: compute_fitness(*g, target) for g in genotypes}
    dist = {}
    for f in fitness.values():
        dist[f] = dist.get(f, 0) + 1
    print("Fitness distribution:", dict(sorted(dist.items())))

    G = build_graph(genotypes, g0_pairs, g1_pairs, outputs)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    plot(G, fitness)


if __name__ == '__main__':
    main()
