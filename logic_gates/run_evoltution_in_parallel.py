from collections import Counter
import concurrent.futures

def run_in_parallel_same_start(function, num_itter, size, isomorphism_counter, *args):
    total_times = []
    total_mutations = []
    total_final_distance = []
    total_mutations_counter = Counter()
    big_fitness_list = []
    big_num_computing_list = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_results = [executor.submit(function, *args) for _ in range(num_itter)]
        for f in concurrent.futures.as_completed(future_results):
            mutation_times, circuit, fitness, mutation_counter, num_computing, final_distance = f.result()
            big_fitness_list.append(fitness)
            big_num_computing_list.append(num_computing)
            total_mutations.append(len(mutation_times))
            total_final_distance.append(final_distance)
            total_times.append(mutation_times[-1])
            if fitness[-1] == 1:
                total_mutations_counter += mutation_counter
                isomorphism_counter.add(circuit, size)
            else:
                print("uh oh didnt get to full fitness")
        return total_mutations, total_times, total_final_distance, {key: value / num_itter for key, value in
                                              total_mutations_counter.items()}, big_fitness_list, big_num_computing_list
