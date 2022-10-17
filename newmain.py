from logic_gates import run_evo, Circuit
import matplotlib.pyplot as plt
import numpy as np

def goal_function(x):
    return x[0] and x[1]

def construct_genome(size):
    genome = []
    for i in range(size):
        genome += [0, 1]
    genome += [2]
    return genome

if __name__ == "__main__":
    initial_circuit = Circuit(2, construct_genome(4))
    total_times = []
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    for i in range(10):
        time, circuits, fitness = run_evo(goal_function, 1000, 0.1, initial_circuit, 1000000)
        print(len(time))
        total_times.append(len(time))
        print(fitness)
        ax.plot(fitness)
    plt.show()
    total_times_array = np.array(total_times)
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.hist(total_times, 20)
    fig.savefig('histogram_fixation_no_loops.pdf', dpi=1200)
    print("Mean:" + str(total_times_array.mean()))
    print("STD:" + str(total_times_array.std()))
    plt.show()
    circuits[0].plot_network(prune=False, filename='unpruned_start.pdf')
    plt.show()
    circuits[-1].plot_network(prune=False, filename='unpruned_end.pdf')
    plt.show()