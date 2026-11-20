"""
Starting genotypes of the bottleneck note, in Circuit / FastCircuit genome format
(nodes 0, 1 = I1, I2; gates = nodes 2 .. N+1; last entry = output node).
"""
import math


def module_length(N: int, alpha: float) -> int:
    """
    Chain-module length L ~ N^alpha. alpha = 0 gives L = 1 (parallel network).
    Rule: round N^alpha half-up, bump to the next odd integer, at least 3.
    This reproduces every module size listed under Table 2 of the note.
    """
    if alpha == 0:
        return 1
    L = int(math.floor(N ** alpha + 0.5))
    if L % 2 == 0:
        L += 1
    return min(max(L, 3), N if N % 2 else N - 1)


def interpolating(N: int, L: int) -> list:
    """
    Chain module of L gates (gate 1 = NAND(I1, I2), gates 2..L = NOT of the
    previous gate), output reads gate L; the other N - L gates are pristine
    NAND(I1, I2) copies. L = 1 is the parallel network, L = N the chain.
    """
    assert L % 2 == 1 and 1 <= L <= N
    genome = [0, 1]                       # chain gate 1 (node 2)
    for j in range(2, L + 1):             # chain gate j = node j + 1
        genome += [j, j]                  # both inputs from chain gate j-1 (node j)
    genome += [0, 1] * (N - L)            # reservoir
    genome += [L + 1]                     # output reads chain gate L
    return genome


def parallel(N: int) -> list:
    return interpolating(N, 1)


def chain(N: int) -> list:
    return interpolating(N, N)


def bottleneck_state(N: int, L: int, k: int) -> list:
    """
    T_k: the interpolating start with chain gate k (odd, >= 3) rewired to take
    one input from a reservoir copy, so it computes NAND(AND, NAND) = TRUE.
    """
    assert k % 2 == 1 and 3 <= k <= L < N
    g = interpolating(N, L)
    g[2 * (k - 1)] = L + 2                # first reservoir gate
    return g
