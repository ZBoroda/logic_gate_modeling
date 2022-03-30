from abc import abstractmethod, ABC
import networkx as nx
import random
import matplotlib.pyplot as plt


def nand(arg1, arg2):
    """
    Performs a nand (not and) operation on two booleans
    :param arg1: boolean 1
    :param arg2: boolean 2
    :return: not (arg1 and arg2)
    """
    return not (arg1 and arg2)


class ContainsLoop(Exception):
    """
    An exception for use in the gate classes in order to truncate the evaluation of a network if there is a back edge
    (an edge that points to a node that has already been visited)
    """
    pass


class Gate(ABC):
    """
    Gates form the backbone of our circuit. This is an abstract class
    """
    def __init__(self):
        self.passed = False
        self.evaluated = False
        self.value = False

    def reset(self):
        self.passed = False
        self.evaluated = False
        self.value = False

    @abstractmethod
    def get_value(self):
        raise NotImplementedError


class InputGate(Gate):
    '''
    An input gate functions as an input
    '''

    def __init__(self):
        self.value = False

    def set_value(self, value):
        self.value = value

    def get_value(self):
        self.evaluated = True
        return self.value


class NandGate(Gate):
    '''
    Nand gate functions as a nan, when get value is called this returns the nand of the values of its two input gates
    '''

    def __init__(self, allow_loops):
        self.allow_loops = allow_loops
        self.gate1 = None
        self.gate2 = None

    def set_inputs(self, gate1, gate2):
        self.gate1 = gate1
        self.gate2 = gate2

    def get_value(self):
        if not self.allow_loops:
            if self.passed and not self.evaluated:
                print('raised')
                raise ContainsLoop()
        if self.passed:
            return self.value
        self.passed = True
        self.value = nand(self.gate1.get_value(), self.gate2.get_value())
        self.evaluated = True
        return self.value


class Circuit:
    """
    Circuit is the object that forms the bedrock of the entire project
    Each Circuit contains a genome and a number of inputs
    The genome is a list of integers: for any nand gate, x, in the circuit with index, i, the integers in the genome at
    location 2*i and 2*i + 1 are the indexes of its input gates. The final integer in the genome is the index of the output gate.
    """

    def __init__(self, num_inputs, genome, allow_loops=True):
        self.allow_loops = allow_loops
        self.num_inputs = num_inputs
        self.genome = genome
        self.inputs = None
        self.gates = None
        self.nand_gates = None
        self.fitness = 0
        self.graph = None

    def construct_circuit(self):
        """Constructs the circuit based on the genome"""
        self.inputs = [InputGate() for _ in range(self.num_inputs)]
        self.nand_gates = [NandGate(self.allow_loops) for _ in range((len(self.genome) - 1) // 2)]
        self.gates = self.inputs + self.nand_gates
        for i in range((len(self.genome) - 1) // 2):
            self.nand_gates[i].set_inputs(self.gates[self.genome[2 * i]], self.gates[self.genome[2 * i + 1]])

    def evaluate(self, in_vals):
        """
        Evaluates truth value of the circuit for specific inputs
        :param in_vals: the values of the inputs
        :return: the truth value of the circuit
        """
        for gate in self.gates:
            gate.reset()
        for i in range(len(self.inputs)):
            self.inputs[i].set_value(in_vals[i])
        return self.gates[self.genome[-1]].get_value()

    def evaluate_expression(self, expression):
        """
        For a expression passed in as a function compares for all possible combinations of inputs how close the
        resulting input from the expression matches the one provided by the circuit
        :param expression:  A boolean expression passed in as a function
        :return: How many matches
        """
        correct_counter = 0
        for i in range(2 ** self.num_inputs):
            binary = format(i, '0' + str(self.num_inputs) + 'b')
            in_vals = [int(digit) == 1 for digit in binary]
            try:
                if self.evaluate(in_vals) == expression(in_vals):
                    correct_counter += 1
            except ContainsLoop:
                self.fitness = 0
                return self.fitness
        self.fitness = correct_counter / (2 ** self.num_inputs)
        if self.fitness == 1.0:
            return 1.0
        useless_gates = 0
        for gate in self.gates:
            if not gate.evaluated:
                # useless_gates += 1
                pass
        self.fitness -= 0.005 * useless_gates
        return self.fitness

    def to_networkx_graph(self, prune=False):
        """
        Builds a simple networkx graph for this circuit
        :return: a networkx graph representation of this circuit
        """
        adj_list = []
        for i in range(self.num_inputs):
            adj_list.append(str(i))
        for i in range((len(self.genome) - 1) // 2):
            adj_list.append(
                str(i + self.num_inputs) + " " + str(self.genome[2 * i]) + " " + str(self.genome[2 * i + 1]))
        print(adj_list)
        g = nx.parse_adjlist(adj_list, create_using=nx.DiGraph, nodetype=int)
        g = g.reverse()
        attributes = {}
        self.evaluate([True] * self.num_inputs)
        for i in range(len(self.gates)):
            n = self.gates[i]
            if n in self.inputs:
                if n.evaluated:
                    attributes[i] = 'used_input'
                else:
                    attributes[i] = 'unused_input'
            elif i == self.genome[-1]:
                attributes[i] = 'output'
            else:
                if n.evaluated:
                    attributes[i] = 'used_nand'
                else:
                    attributes[i] = 'unused_nand'
        nx.set_node_attributes(g, attributes, name="type")
        if prune:
            color_dict = {'unused_input': 0, 'unused_nand': 1, 'used_input': 2, 'used_nand': 3, 'output': 4}
            plotable_nodes = [n for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            # plotable_edges = [e for e in g.edges() if e[1] in plotable_nodes]
            g = g.subgraph(plotable_nodes)
        self.graph = g
        return g

    def plot_network(self, prune=False):
        color_dict = {'unused_input': 0, 'unused_nand': 1, 'used_input': 2, 'used_nand': 3, 'output': 4}
        g = self.to_networkx_graph(prune)
        # colors = [color_dict[g.nodes[str(i)]["type"]] for i in range(len(g.nodes()))]
        if not prune:
            colors = [color_dict[g.nodes[n]["type"]] for n in g.nodes()]
            nx.draw_networkx(g, with_labels=True, node_color=colors, cmap=plt.get_cmap('cool'), font_color='white')
        else:
            plotable_nodes = [n for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            colors = [color_dict[g.nodes[n]["type"]] for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            plotable_edges = [e for e in g.edges() if e[1] in plotable_nodes]
            nx.draw_networkx(g, node_color=colors, cmap=plt.get_cmap('Dark2'), font_color='white',
                             nodelist=plotable_nodes, edgelist=plotable_edges, with_labels=True)
        plt.show()

    def duplicate(self):
        copy = Circuit(self.num_inputs, self.genome[:], self.allow_loops)
        copy.construct_circuit()
        return copy

    def mutate(self, probability=0.8):
        if random.random() > probability:
            return
        possible_mutations = [self.point_mutation,
                              self.point_mutation,
                              # self.point_mutation,
                              self.gene_duplication,
                              # self.gene_deletion,
                              self.gene_deletion,
                              self.gene_deletion,
                              # self.gene_duplication,
                              self.gene_addition
                              ]
        random.choice(possible_mutations)()
        self.construct_circuit()
        # Makes a mutation Question: Do I want to change the number of mutations based on the size of the genome
        # for i in range(1 + int(len(self.genome) / 20)):
        # print(len(self.genome))

    def point_mutation(self):
        # print('point')
        index = random.randrange(len(self.genome))
        new_value = random.randrange(len(self.gates))
        self.genome[index] = new_value

    def gene_duplication(self):
        # print('dup')
        if len(self.nand_gates) == 0:
            return
        gene_index = random.randrange(len(self.nand_gates))
        gene = self.genome[2 * gene_index], self.genome[2 * gene_index + 1]
        output = self.genome[-1]
        self.genome[-1:] = gene
        self.genome.append(output)

    def gene_deletion(self):
        # print('del')
        if len(self.nand_gates) == 0:
            return
        gene_index = random.randrange(len(self.nand_gates))
        gene = self.genome[2 * gene_index], self.genome[2 * gene_index + 1]
        real_index = gene_index + self.num_inputs
        for i in range(len(self.genome)):
            if self.genome[i] == real_index:
                if (gene[0] == real_index) and (gene[1] == real_index):
                    self.genome[i] = i // 2
                elif gene[0] == real_index:
                    self.genome[i] = gene[1]
                elif gene[1] == real_index:
                    self.genome[i] = gene[0]
                else:
                    self.genome[i] = random.choice(gene)
            if self.genome[i] > real_index:
                self.genome[i] = self.genome[i] - 1
        self.genome.pop(2 * gene_index)
        self.genome.pop(2 * gene_index + 1)

    def gene_addition(self):
        # print('add')
        gene = random.randrange(len(self.gates)), random.randrange(len(self.gates))
        output = self.genome[-1]
        self.genome[-1:] = gene
        self.genome.append(output)
