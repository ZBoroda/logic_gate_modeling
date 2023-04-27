from logic_gates import run_evolution_strong_selection, run_random_walk, Circuit
import pickle
from multiprocessing import Pool


def worker(size, fitness_function, function_name, num_itter):
    genomes = set()
    genome = [0, 0] * size + [0]
    for i in range(num_itter):
        c = Circuit(2, genome)
        if c.evaluate_expression(fitness_function) == 0.0:
            genomes.add(tuple(genome))
            if len(genomes) % 100 == 0:
                print(size, len(genomes))
        c.mutate()
        genome = c.genome
    list_genomes = []
    for g in genomes:
        list_genomes.append(list(g))
    return list_genomes

def generate_zero_fit_genomes(sizes, fitness_function, function_name, num_itter=1000000):

    with Pool() as pool:
        results = pool.starmap(worker, [(size, fitness_function, function_name, num_itter) for size in sizes])

    for result, size in zip(results, sizes):
        with open('zero_fitness_genome_files/' + function_name + '/size_' + str(size) + '.pickle', 'wb') as f:
            pickle.dump(result, f)
