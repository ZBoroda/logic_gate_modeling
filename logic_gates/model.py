from abc import abstractmethod, ABC
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from logic_gates.gates import Circuit
from multiprocessing import Pool


class Model:
    def __init__(self, num_circuits, num_inputs, initial_genome):
        self.num_circuits = num_circuits
        self.circuits = [Circuit(num_inputs, initial_genome[:], allow_loops=False) for _ in range(self.num_circuits)]
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
        pool = Pool()
        for i in range(max_gens):
            fitness.append(self.circuits[0].evaluate_expression(goal))
            print('\t', i, fitness[i])
            self.circuits = pool.map(self.perform_evolution_step, [circuit for circuit in self.circuits])
            self.circuits.sort(key=lambda circuit: circuit.fitness, reverse=True)
            print()
            if i >= 1:
                if self.circuits[0].fitness == 1.0 and self.circuits[30].fitness == 1.0:
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
            if i % 10 == 0:
                self.circuits[0].plot_network(prune=False)
        fitness.append(self.circuits[0].fitness)
        print('\t', i + 1, fitness[i])
        return fitness, self.circuits
