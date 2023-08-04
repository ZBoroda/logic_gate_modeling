from collections import Counter
from scipy.spatial.distance import hamming
import numpy as np

from logic_gates import Circuit

def find_clusters_rw(f, x_init: Circuit, m_max: int, genome_size: int):
    """
    Run evolutionary dynanamics in strong selection weak mutation regime.

    Inputs:
      f - fitness landscape, must be function that takes array of length x_init
      x_init - array of length L specifying the initial genome
      m_max - the number of mutations to run the simulation
      genome_size - the size of the genome running with, this is just for recording of results
    """
    size = 0
    num_cluster_sizes = 0
    sum_cluster_sizes = 0
    sum_cluster_sizes_squared = 0
    sum_cluster_sizes_hamming = 0
    sum_cluster_sizes_hamming_squared = 0
    gap = 0
    num_cluster_gaps = 0
    sum_cluster_gaps = 0
    sum_cluster_gaps_squared = 0
    sum_cluster_gaps_hamming = 0
    sum_cluster_gaps_hamming_squared = 0
    cluster_gaps = []
    x = x_init.duplicate()
    first_genome = []
    last_genome = []
    for m in range(m_max):
        #if m % 1000 == 0:
        #    print(genome_size, m, num_cluster_sizes, num_cluster_gaps)
        fittness = x.evaluate_expression(f)
        if fittness == 1:
            size += 1
            if not gap == 0:
                first_genome = x.genome
                if not last_genome == []:
                    gap_hamming = hamming(first_genome, last_genome) * len(first_genome)
                    sum_cluster_gaps_hamming += gap_hamming
                    sum_cluster_gaps_hamming_squared += gap_hamming * gap_hamming
                sum_cluster_gaps += gap
                sum_cluster_gaps_squared += gap * gap
                num_cluster_gaps += 1
                gap = 0
            last_genome = x.genome
        else:
            gap += 1
            if not size == 0:
                if not first_genome == []:
                    gap_hamming = hamming(first_genome, last_genome) * len(first_genome)
                    sum_cluster_sizes_hamming += gap_hamming
                    sum_cluster_sizes_hamming_squared += gap_hamming * gap_hamming
                sum_cluster_sizes += size
                sum_cluster_sizes_squared += size * size
                size = 0
                num_cluster_sizes += 1
        x.mutate()
    cluster_sizes = {'mean': sum_cluster_sizes / num_cluster_sizes}
    cluster_sizes['var'] = (sum_cluster_sizes_squared / num_cluster_sizes) - (cluster_sizes['mean'] ** 2)
    cluster_gaps = {'mean': sum_cluster_gaps / num_cluster_gaps}
    cluster_gaps['var'] = (sum_cluster_gaps_squared / num_cluster_gaps) - (cluster_gaps['mean'] ** 2)

    cluster_sizes_hamming = {'mean': sum_cluster_sizes_hamming / num_cluster_sizes}
    cluster_sizes_hamming['var'] = (sum_cluster_sizes_hamming_squared / num_cluster_sizes) - (cluster_sizes_hamming['mean'] ** 2)
    cluster_gaps_hamming = {'mean': sum_cluster_gaps_hamming / num_cluster_gaps}
    cluster_gaps_hamming['var'] = (sum_cluster_gaps_hamming_squared / num_cluster_gaps) - (cluster_gaps_hamming['mean'] ** 2)
    return cluster_sizes, cluster_gaps, cluster_sizes_hamming, cluster_gaps_hamming, genome_size


