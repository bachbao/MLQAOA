import os
import numpy as np


from itertools import combinations, chain
# import sys

from scipy.spatial import distance_matrix
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

import networkx as nx
import json
from itertools import product
from copy import copy




def powerset(iterable):
    """Returns the powerset of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, q) for q in range(len(s) + 1))


def is_independent(graph, solution):
    """Checks if a given solution is independent."""
    if isinstance(solution, dict):
        sorted_keys = sorted(solution.keys())
        solution = np.where([int(solution[i] > 0) for i in sorted_keys])[0]
    for edge in product(solution, repeat=2):
        if graph.has_edge(*edge):
            return False
    return True

def count_violations(graph, solution):
    nvio = 0
    for edge in product(solution, repeat=2):
        if graph.has_edge(*edge):
            nvio += 1
    return nvio


def find_mis(graph, maximum=True):
    """Finds a maximal independent set of a graph, and returns its bitstrings."""
    if maximum is False:
        colored_nodes = nx.maximal_independent_set(graph)
        return len(colored_nodes), colored_nodes
    else:
        solutions = []
        maximum = 0
        for subset in powerset(graph.nodes):
            if is_independent(graph, subset):
                if len(subset) > maximum:
                    solutions = [subset]
                    maximum = len(subset)
                elif len(subset) == maximum:
                    solutions.append(subset)
        return maximum, solutions
    
    
def find_maxcut(graph): 
    """Find the maximum cut assignment for a graph with size < nc using brute force and return its bitstring."""
    node_set = graph.nodes
    nc = graph.number_of_nodes()
    node_index_dict = {node: index for index, node in enumerate(node_set)}
    max_cost = float('-inf')
    solutions = []

    for r in range(1, nc + 1):
        # Generate all possible combinations of nodes to form the cut
        cut_combinations = combinations(node_set, r)

        for cut_nodes in cut_combinations:
            bit_string = [1 if node in cut_nodes else 0 for node in node_set]
            cost = MaxCut_cost(bit_string, graph, node_index_dict)

            if cost > max_cost:
                solutions = [{node: bit_string[index] for index, node in enumerate(node_set)}]
                max_cost = cost
            elif cost == max_cost:
                solutions.append({node: bit_string[index] for index, node in enumerate(node_set)})

    return max_cost, solutions

def MaxCut_cost(bit_string,graph,node_index_dict):
    """ Evaluate the MaxCut cost for the given assignment """
    cost = 0 
    # The cost related to node sum_i w_i x_i (disable this if you do not want self-loop)
#     for node in graph.nodes(): 
#         index_i = node_index_dict[node] 
#         weight_i = graph[node][node]['weight'] 
#         cost += weight_i*bit_string[index_i] 

    # The cost related to edge sum_ij w_ij x_i(1-x_j) = sum_(i,j) w_ij[ x_i(1-x_j) + x_j(1-x_i)] 
    for edge in graph.edges(): 
        node_i = edge[0]
        node_j = edge[1]
        index_i = node_index_dict[node_i]
        index_j = node_index_dict[node_j] 
        weight_ij = graph[node_i][node_j]['weight']
        cost += weight_ij*bit_string[index_i]*(1-bit_string[index_j]) + weight_ij*(1-bit_string[index_i])*bit_string[index_j] 
    return cost 


def bit_string_generator(nc): 
    """ This function is used to create 2^nc bit string where instead of 0 and 1, we use -1 and 1 """  
    string_list = []
    for i in range(2 ** nc):
        bitstring = [int(x) for x in format(i, f'0{nc}b')]
        string_list.append(bitstring)
    return string_list 


def vertices_to_graph(vertices, radius=7.5e-6):
    """Converts the positions of vertices into a UDG"""
    dmat = distance_matrix(vertices, vertices)
    adj = (dmat < radius).astype(int) - np.eye(len(vertices))
    # zr = np.where(~np.any(adj, axis=0))
    # adj = np.delete(np.delete(adj, zr, 1), zr, 0)
    return nx.from_numpy_array(adj)



def get_mis_size(graph, solution):
    print(f"Set is independent? {is_independent(graph, solution)}")
    if isinstance(solution, dict):
        return np.count_nonzero([int(i > 0) for i in solution.values()])
    else:
        return len(solution)
