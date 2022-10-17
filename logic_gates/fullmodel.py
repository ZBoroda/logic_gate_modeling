from abc import abstractmethod, ABC
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from logic_gates.gates import Circuit
from multiprocessing import Pool
from tqdm import tqdm
import typing


class FullModel:
    """
    A Model is an object that contains a population of num_circuits circuits each with num_inputs inputs
    """

    def __init__(self, num_circuits, num_inputs, initial_genome, mutation_probability=0.2,
                 mutation_proportional_to_size=True):
        """
        Constructs a model object with a population of num_circuits circuits each with num_inputs inputs and genome
        initial_genome

        :param num_circuits: the number of circuits in the population
        :param num_inputs: the number of inputs for each circuit
        :param initial_genome: the initial genome of each circuit
        :param mutation_probability: the probability of there being a mutation
        """
        self.num_circuits = num_circuits
        self.circuits = [Circuit(num_inputs, initial_genome[:], allow_loops=False) for _ in range(self.num_circuits)]
        self.mutation_probability = mutation_probability
        for circuit in self.circuits:
            circuit.construct_circuit()
        self.goal = None
        self.mutation_proportional_to_size = mutation_proportional_to_size

    def perform_evolution_step(self, circuit):
        """
        Function used for multiprocessing so that I dont need to deal with shared memory.
        Performs an evolution step on a circuit: mutates and then evaluates the fitness returning the mutated circuit

        :param circuit: the circuit being mutated
        :return: this mutated circuit
        """
        circuit.mutate(probability=self.mutation_probability,
                       mutation_rate_prop_to_size=self.mutation_proportional_to_size)
        circuit.evaluate_expression(self.goal)
        return circuit

    def perform_selection_top_50(self):
        self.circuits.sort(key=lambda circuit: circuit.fitness, reverse=True)
        if self.num_circuits % 2 == 0:
            self.circuits[self.num_circuits // 2:] = [circuit.duplicate() for circuit in
                                                      self.circuits[:self.num_circuits // 2]]
        else:
            self.circuits[self.num_circuits // 2:] = [circuit.duplicate() for circuit in
                                                      self.circuits[:self.num_circuits // 2 + 1]]

    def perform_selection_wright_fisher(self):
        new_pool = self.circuits[:]
        new_pool += [circuit.duplicate() for circuit in self.circuits for _ in range(int(circuit.fitness * 4))]
        self.circuits = random.sample(new_pool, self.num_circuits)

    def evolve(self, goal, max_gens: int = 500, debug=False) -> typing.Tuple[
        typing.List[int], typing.List[Circuit], typing.List[int], int]:
        """
        Performs evolution for max gens generations toward goal goal

        :param goal: the desired logic function that is being evolved towards
        :param max_gens: the number of generations to run this experiment for
        :param debug: prints the number of
        :return: the fitness trajectory of the top circuit, the final list of circuits
        """
        self.goal = goal
        fitness_top = []
        fitness_sum = []
        # first_full_fitness = float("inf")
        first_full_fitness = max_gens
        for i in range(max_gens):
            random.shuffle(self.circuits)
            self.circuits = list(map(self.perform_evolution_step, self.circuits))
            fitness_top.append(max(circuit.fitness for circuit in self.circuits))
            fitness_sum.append(np.mean(np.array([circuit.fitness for circuit in self.circuits])))
            if debug:
                print('\t', i, fitness_top[i], fitness_sum[i])
                print()
            if fitness_top[i] == 1.0:
                if fitness_top[i - 1] != 1.0:
                    first_full_fitness = i
            if i >= 1:
                if fitness_top[i] == 1.0 and fitness_sum[i] == 1.0:
                    return fitness_top, self.circuits, fitness_sum, first_full_fitness
                if debug and fitness_top[i] < fitness_top[i - 1]:
                    print('Oh no')
            self.perform_selection_wright_fisher()
        return fitness_top, self.circuits, fitness_sum, first_full_fitness
