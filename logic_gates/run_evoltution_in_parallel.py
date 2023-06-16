from collections import Counter
import concurrent.futures


def run_in_parallel_same_start(function, num_itter, size, isomorphism_counter, *args):
    total_times = []
    total_mutations = []
    total_mutations_counter = Counter()
    big_fitness_list = []
    big_num_computing_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = [executor.submit(function, *args) for _ in range(num_itter)]
        for f in concurrent.futures.as_completed(future_results):
            mutation_times, circuit, fitness, mutation_counter, num_computing = f.result()
            big_fitness_list.append(fitness)
            big_num_computing_list.append(num_computing)
            if fitness[-1] == 1:
                total_mutations.append(len(mutation_times))
                total_times.append(mutation_times[-1])
                isomorphism_counter.add(circuit, size)
                total_mutations_counter += mutation_counter
            else:
                print("uh oh didnt get to full fitness")
        return total_mutations, total_times, {key: value / num_itter for key, value in
                                              total_mutations_counter.items()}, big_fitness_list, big_num_computing_list
