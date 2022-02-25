from abc import abstractmethod, ABC
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from gates import Circuit
from multiprocessing import Pool


class Model:
    def __init__(self, num_circuits, num_inputs, initial_genome):
        self.num_circuits = num_circuits
        self.circuits = [Circuit(num_inputs, initial_genome[:]) for _ in range(self.num_circuits)]
        for circuit in self.circuits:
            circuit.construct_circuit()
        self.goal = None

    def perform_evolution_step(self, circuit):
        circuit.mutate()
        circuit.evaluate_expression(self.goal)
        return circuit

    def evolve(self, goal, max_gens=500):
        self.goal = goal
        fitness = []
        pool = Pool(8)
        for i in range(max_gens):
            fitness.append(self.circuits[0].evaluate_expression(goal))
            print('\t', i, fitness[i])
            self.circuits = pool.map(self.perform_evolution_step, [circuit for circuit in self.circuits])
            self.circuits.sort(key=lambda circuit: circuit.fitness, reverse=True)
            if i >= 1:
                if self.circuits[30].fitness == 1.0:
                    fitness.append(self.circuits[0].fitness)
                    print('\t', i + 1, fitness[i])
                    return fitness, self.circuits
                if fitness[i] < fitness[i - 1]:
                    print('Oh no')
            if self.num_circuits % 2 == 0:
                self.circuits[self.num_circuits // 2:] = [circuit.duplicate() for circuit in
                                                          self.circuits[:self.num_circuits // 2]]
            else:
                self.circuits[self.num_circuits // 2:] = [circuit.duplicate() for circuit in
                                                          self.circuits[:self.num_circuits // 2 + 1]]
            random.shuffle(self.circuits)
            if i == 1:
                print('hi')
            if i % 2000 == 0:
                self.circuits[0].plot_network(prune=False)
        fitness.append(self.circuits[0].fitness)
        print('\t', i + 1, fitness[i])
        return fitness, self.circuits


def function_goal(x):
    return not (x[0] or x[1]) #and (x[0] or x[1]) and (x[2] or x[3]) and not (x[2] and x[3])


if __name__ == "__main__":
    '''
    c = Circuit(2, [0, 1, 2])
    c.construct_circuit()
    print(c.evaluate_expression(lambda arr: not (arr[0] and arr[1])))
    c = Circuit(4, [0, 1, 1, 2, 4, 5, 5, 3, 6, 7, 8])
    c.construct_circuit()
    c.plot_network()
    print(c.evaluate_expression(
        lambda arr: nand(nand(nand(arr[0], arr[1]), nand(arr[1], arr[2])), nand(nand(arr[1], arr[2]), arr[3]))))
    c.mutate()
    print(c.evaluate_expression(
        lambda arr: nand(nand(nand(arr[0], arr[1]), nand(arr[1], arr[2])), nand(nand(arr[1], arr[2]), arr[3]))))
    '''
    c = Circuit(2, [0, 1, 2])
    c.construct_circuit()
    # c.plot_network()
    m = Model(200, 2, [0, 1, 2])
    fitness, circuits = m.evolve(function_goal, max_gens=3000)  # lambda arr: (arr[0] or arr[1]))
    plt.plot(fitness)
    #print('hi')
    plt.show()
    # nx.draw(c.to_networkx_graph())
    # plt.draw()
    # print("hi")
    circuits[0].plot_network()
    #print('hi')
