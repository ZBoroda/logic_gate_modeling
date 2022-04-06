from abc import abstractmethod, ABC
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from logic_gates.gates import Circuit
from multiprocessing import Pool


class Model:
    """
    A Model is an object that contains a population of num_circuits circuits each with num_inputs inputs
    """
    def __init__(self, num_circuits, num_inputs, initial_genome):
        """
        Constructs a model object with a population of num_circuits circuits each with num_inputs inputs and genome
        initial_genome

        :param num_circuits: the number of circuits in the population
        :param num_inputs: the number of inputs for each circuit
        :param initial_genome: the initial genome of each circuit
        """
        self.num_circuits = num_circuits
        self.circuits = [Circuit(num_inputs, initial_genome[:], allow_loops=False) for _ in range(self.num_circuits)]
        for circuit in self.circuits:
            circuit.construct_circuit()
        self.goal = None

    def perform_evolution_step(self, circuit):
        """
        Function used for multiprocessing so that I dont need to deal with shared memory.
        Performs an evolution step on a circuit: mutates and then evaluates the fitness returning the mutated circuit

        :param circuit: the circuit being mutated
        :return: this mutated circuit
        """
        circuit.mutate()
        circuit.evaluate_expression(self.goal)
        return circuit

    def evolve(self, goal, max_gens=500):
        """
        Performs evolution for max gens generations toward goal goal

        :param goal: the desired logic function that is being evolved towards
        :param max_gens: the number of generations to run this experiment for
        :return: the fitness trajectory of the top circuit, the final list of circuits
        """
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
