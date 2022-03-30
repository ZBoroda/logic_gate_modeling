from logic_gates import Circuit
from logic_gates import Model

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt


def function_goal(x):
    return (x[0] and x[1])  # and (x[0] or x[1]) and (x[2] or x[3]) and not (x[2] and x[3])


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
    c = Circuit(2, [0, 1, 2, 2])
    c.construct_circuit()
    c.plot_network()
    m = Model(200, 2, [0, 1, 2])
    fitness, circuits = m.evolve(function_goal, max_gens=3000)  # lambda arr: (arr[0] or arr[1]))
    plt.plot(fitness)
    # print('hi')
    plt.show()
    # nx.draw(c.to_networkx_graph())
    # plt.draw()
    # print("hi")
    circuits[0].plot_network(prune=False)
    circuits[0].plot_network(prune=True)
    # print('hi')
