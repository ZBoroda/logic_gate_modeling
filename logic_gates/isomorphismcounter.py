from collections import defaultdict, Counter

'''
Class is still a work in progress I use the networkX library's definition of isomorphism in order to categorize and 
count networks in terms of which other networks they are similar too

A future counter will use a clever trick (ordering the gates in a specific way)
possibly implemented in the circuit class itself in order to make categorization a simpler task,
and make the storing of large networkx objects in RAM unnecessary.
'''


class IsomorphismCounter:
    """
    This object is a wrapper over a default dict of counters. The schema for this internal data structure is:
    dict[network: dict[category: int]] where a class of network, identified by the first network of that class that this
    counter sees maps to a dictionary that describes the class.

    This object splits networks into their equivalence classes with the equivalence relation = isomorphism. For each
    entry into the counter, the counter will categorize the network into its corresponding equivalence class,
    incrementing the counter for that class. It will also record the value of the network's size in the class's size
    dictionary.
    """

    def __init__(self):
        """
        Initialize a new IsomorphismCounter.
        """
        self.counter = defaultdict(Counter)

    def add(self, network, size: int):
        """
        Categorize this network and increment the proper counters based off its equivalence class membership

        :param network:
        :param size:
        :return:
        """
        for key in self.counter:
            if key.is_isomorphic(network, pruned=True):
                self.counter[key][size] += 1
                self.counter[key]["total"] += 1
                return key
        self.counter[network][size] += 1
        self.counter[network]["total"] += 1
        return network

    def get_networks(self):
        return self.counter.keys()

    def get_number_by_size(self, size: int):
        return {key: value[size] for key, value in self.counter.items()}

    def get_number_networks_total(self):
        return {key: value['total'] for key, value in self.counter.items()}
