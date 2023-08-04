from abc import abstractmethod, ABC
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict


def nand(arg1: bool, arg2: bool):
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
    Gates form the backbone of our circuit. This is an abstract class.
    """

    def __init__(self):
        self.passed = False
        self.evaluated = False
        self.value = False
        self.contains_loop = False

    def reset(self):
        """
        Resets the gate to its pre-computation self.
        """
        self.passed = False
        self.evaluated = False
        self.value = False
        self.contains_loop = False

    @abstractmethod
    def get_value(self) -> bool:
        """
        Returns the value stored at this gate.

        :return: the value stored
        """
        raise NotImplementedError


class InputGate(Gate):
    """
    An input gate functions as an input
    Default value is False
    """

    def __init__(self):
        """
        Constructs a new InputGate with default value False
        """
        self.value = False

    def set_value(self, value: bool):
        """
        Sets the value stored at this gate to value

        :param value: A boolean that will be stored at this gate
        """
        self.value = value

    def get_value(self) -> bool:
        """
        Returns the value stored at this gate.

        :return: the value stored
        """
        self.evaluated = True
        return self.value


class NandGate(Gate):
    """
    NandGate functions as a nand, when get value is called this returns the nand of the values of its two input gates

    If allows_loops is True then this nand gate will return False whenever it's getValue method is called by a
    recursive call to getValue originating inside this nandGate, if False this nandGate will raise a ContainsLoop
    exception whenever it's getValue method is called by a recursive call to getValue originating inside this nandGate.
    """

    def __init__(self, allow_loops: bool):
        """
        Constructs a new NandGate

        :param allow_loops: whether this nand gate is allowed to be in a loop.
        """
        self.allow_loops = allow_loops
        self.gate1 = None
        self.gate2 = None

    def set_inputs(self, gate1: Gate, gate2: Gate):
        """
        Set the input gates to gate1 and gate2

        :param gate1: the first input
        :param gate2: the second input
        """
        self.gate1 = gate1
        self.gate2 = gate2

    def get_value(self):
        """
        Returns that value of this node. Value gets calculated by performing nand of the values of the two inputs.
        If allows_loops is True then this nand gate will return False whenever it's getValue method is called by a
        recursive call to getValue originating inside this nandGate, if False this nandGate will raise a ContainsLoop
        exception whenever it's getValue method is called by a recursive call to getValue originating inside this nandGate.

        :return: the value of this node
        """
        if self.contains_loop:
            raise Exception("I dont think I should ever get here")
            raise ContainsLoop()
        if not self.allow_loops:
            # this node is in a loop when this method is called again before it has ever evaluated the nand of its
            # inputs. (passed is true so this method has been called once, but evaluated is false)
            if self.passed and not self.evaluated:
                # print('raised')
                self.contains_loop = True
                raise ContainsLoop()
        if self.passed:
            return self.value
        self.passed = True
        try:
            self.value = nand(self.gate1.get_value(), self.gate2.get_value())
        except ContainsLoop:
            self.contains_loop = True
            raise ContainsLoop()
        self.evaluated = True
        return self.value


class Circuit:
    """
    Circuit is the object that forms the bedrock of the entire project.

    Each Circuit contains a genome and a number of inputs

    The genome is a list of integers: for any nand gate, x, in the circuit with index, i, the integers in the
    genome at location 2*i and 2*i + 1 are the indexes of its input gates. The final integer in the genome is the
    index of the output gate.
    """

    def __init__(self, num_inputs, genome, allow_loops: bool = False, use_dp: bool = False):
        self.allow_loops = allow_loops
        self.num_inputs = num_inputs
        self.genome = genome
        self.inputs = None
        self.gates = None
        self.nand_gates = None
        self.fitness = 0
        self.graph = None
        self.use_dp = use_dp
        if self.use_dp:
            self.dynamic_programming_dict = {}
        self.construct_circuit()
        self.evaluated = False
        self.evaluated_dict = {}

    def construct_circuit(self):
        """
        Constructs the circuit based on the genome
        """
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

    def evaluate_expression(self, expression, penalize_useless: bool = False) -> float:
        """
        For an expression passed in as a function compares for all possible combinations of inputs how close the
        resulting input from the expression matches the one provided by the circuit

        :param expression:  A boolean expression passed in as a function
        :param penalize_useless: Whether to penalize useless gates (gates that do not affect fitness)
        :return: the percentage of matches between this network and the expression
        """
        if self.evaluated:
            return self.evaluated_dict[expression]
        if self.use_dp:
            if str(self.genome) in self.dynamic_programming_dict:
                self.fitness = self.dynamic_programming_dict[str(self.genome)]
                return self.fitness
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
        if penalize_useless:
            if self.fitness == 1.0:
                return 1.0
            useless_gates = 0
            for gate in self.gates:
                if not gate.evaluated:
                    useless_gates += 1
            self.fitness -= 0.005 * useless_gates
        if self.use_dp:
            if str(self.genome) not in self.dynamic_programming_dict:
                self.dynamic_programming_dict[str(self.genome)] = self.fitness
        self.evaluated = True
        self.evaluated_dict[expression] = self.fitness
        return self.fitness

    def contains_loops(self):
        '''
        Checks if this network has a loop (a cycle). This check is performed by for each node checking if the network
        based in that node has a cycle

        :return: True if network has a cycle
        '''
        for gate in self.gates:
            gate.reset()
        for gate in self.gates:
            try:
                gate.get_value()
            except ContainsLoop:
                return True
        return False

    def num_gates_computing(self) -> int:
        self.evaluate([True] * self.num_inputs)
        cnt = 0
        for gate in self.gates:
            if gate.evaluated:
                cnt += 1
        return cnt

    def to_networkx_graph(self, prune: bool = False) -> nx.DiGraph:
        """
        Builds a simple networkx graph for this circuit

        :param prune: if true only return the part of the network used in computation of the expression
        :return: a networkx graph representation of this circuit
        """
        adj_list = []
        for i in range(self.num_inputs):
            adj_list.append(str(i))
        for i in range((len(self.genome) - 1) // 2):
            adj_list.append(
                str(i + self.num_inputs) + " " + str(self.genome[2 * i]) + " " + str(self.genome[2 * i + 1]))
        # print(adj_list)
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
        self.graph = g
        if prune:
            color_dict = {'unused_input': 0, 'unused_nand': 1, 'used_input': 2, 'used_nand': 3, 'output': 4}
            plotable_nodes = [n for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            # plotable_edges = [e for e in g.edges() if e[1] in plotable_nodes]
            g = g.subgraph(plotable_nodes)
        return g

    def is_isomorphic(self, network, pruned: bool = True) -> bool:
        graph = self.to_networkx_graph(pruned)
        other_graph = network.to_networkx_graph(pruned)
        return nx.is_isomorphic(graph, other_graph)

    def plot_network(self, prune: bool = False, filename=None):
        """
        Plot the network

        :param prune: if true then only plot the part of the network used in computation of the expression
        """
        plt.subplot()
        color_dict = {'unused_input': 0, 'unused_nand': 1, 'used_input': 2, 'used_nand': 3, 'output': 4}
        g = self.to_networkx_graph(prune)
        colors = [color_dict[g.nodes[n]["type"]] for n in g.nodes()]
        # colors = [color_dict[g.nodes[str(i)]["type"]] for i in range(len(g.nodes()))]
        if not prune:
            # colors = [color_dict[g.nodes[n]["type"]] for n in g.nodes()]
            nx.draw_networkx(g, with_labels=True, node_color=colors, cmap=plt.get_cmap('cool'), font_color='white')
        else:
            '''
            plotable_nodes = [n for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            colors = [color_dict[g.nodes[n]["type"]] for n in g.nodes() if color_dict[g.nodes[n]["type"]] > 1]
            plotable_edges = [e for e in g.edges() if e[1] in plotable_nodes]
            nx.draw_networkx(g, node_color=colors, cmap=plt.get_cmap('Dark2'), font_color='white',
                             nodelist=plotable_nodes, edgelist=plotable_edges, with_labels=True)
            '''
            nx.draw_networkx(g, with_labels=True, node_color=colors, cmap=plt.get_cmap('Dark2'), font_color='white')
        if not filename:
            plt.show()
        else:
            plt.savefig(filename, dpi=1000)
            
    def duplicate(self):
        """
        Returns a copy of this circuit

        :return:  copy of this circuit
        """
        copy = Circuit(self.num_inputs, self.genome[:], self.allow_loops)
        copy.construct_circuit()
        return copy

    def mutate(self, probability: float = 1, model_type: str = 'Basic', mutation_rate_prop_to_size: bool = False):
        """
        Performs a mutation on this circuit. There are 4 types of mutations:
        :func:`point_mutation`
        :func:`gene_duplication`
        :func:`gene_deletion`
        :func:`gene_addition`

        :param probability: the probability that a mutation will occur
        :param model_type: the types of mutations that
        :param mutation_rate_prop_to_size: whether the mutation rate should scale with the size
        the network should allow. 'Basic', the default, means only point mutations. 'All' means run with all mutations.
        """
        if mutation_rate_prop_to_size:
            if random.random() > probability * len(self.genome):
                return
        else:
            if random.random() > probability:
                return
        possible_mutations = []
        if model_type == 'Basic':
            possible_mutations = [self.point_mutation]
        else:
            possible_mutations = [self.point_mutation,
                                  self.gene_duplication,
                                  self.gene_deletion,
                                  self.gene_deletion,
                                  self.gene_addition
                                  ]
        old_genome = self.genome.copy()
	#Just to check if im getting stuck in here for extended periods of time
        cnt = 0
        while True:
            cnt += 1
            if cnt % 1000 == 0:
                print("Stuck finding loop less mutation for a while")
            random.choice(possible_mutations)()
            self.construct_circuit()
            if self.allow_loops or not self.contains_loops():
                break
            self.genome = old_genome[:]
        # Makes a mutation Question: Do I want to change the number of mutations based on the size of the genome
        # for i in range(1 + int(len(self.genome) / 20)):
        # print(len(self.genome))

    def point_mutation(self):
        """
        Perform a point mutation. Swap one of the spots in the genome for another gate's numbers.
        """
        # print('point')
        self.construct_circuit()
        if self.contains_loops():
            raise Exception("Big error at the beginning gene already has duplicate" +str(self.genome))
        old_genome = self.genome[:]
        tried_and_failed = defaultdict(set)
        index = random.randrange(len(self.genome))
        if index == len(self.genome) - 1:
            new_value = random.randrange(len(self.nand_gates)) + self.num_inputs
        else:
            new_value = random.randrange(len(self.gates))
        self.genome[index] = new_value
        self.construct_circuit()
        cnt = 0
        while not self.allow_loops and self.contains_loops():
            self.genome = old_genome[:]
            tried_and_failed[index].add(new_value)
            while new_value in tried_and_failed[index]:
                index = random.randrange(len(self.genome))
                if index == len(self.genome) - 1:
                    new_value = random.randrange(len(self.nand_gates)) + self.num_inputs
                else:
                    new_value = random.randrange(len(self.gates))
                cnt+=1
            self.genome[index] = new_value
            self.construct_circuit()
            cnt += 1
        if index == len(self.genome) - 1 or self.nand_gates[index // 2].evaluated:
            self.evaluated = False
        return


    def gene_duplication(self):
        """
        Duplicate a gate, take the coding for a gate 2 letters in the genome and add
        that same gate to the end of the genome
        """
        # print('dup')
        if len(self.nand_gates) == 0:
            return
        gene_index = random.randrange(len(self.nand_gates))
        gene = self.genome[2 * gene_index], self.genome[2 * gene_index + 1]
        output = self.genome[-1]
        self.genome[-1:] = gene
        self.genome.append(output)

    def gene_deletion(self):
        """
        Delete a gene
        """
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
        """
        Add a random gene to the end of the genome
        """
        # print('add')
        gene = random.randrange(len(self.gates)), random.randrange(len(self.gates))
        output = self.genome[-1]
        self.genome[-1:] = gene
        self.genome.append(output)

    def __repr__(self):
        return str(self.genome)

    def __str__(self):
        return 'Circuit ' + str(self.num_inputs) + ' inputs ' + str(self.genome) + ' genome'
