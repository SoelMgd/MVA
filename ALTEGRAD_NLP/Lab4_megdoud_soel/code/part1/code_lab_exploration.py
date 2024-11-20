"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
print('Task 1. \n')

file_path = '/content/ALTEGRAD_LAB4/code/datasets/CA-HepTh.txt'
G = nx.read_edgelist(file_path)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")


############## Task 2
print('\nTask 2. \n')
components = list(nx.connected_components(G))


num_components = len(components)
print(f"Number of connected components: {num_components}")


largest_component = max(components, key=len)
G_largest = G.subgraph(largest_component)

num_nodes_largest = G_largest.number_of_nodes()
num_edges_largest = G_largest.number_of_edges()
print(f"Number of nodes in the biggest connected component: {num_nodes_largest}")
print(f"Number of edges in the biggest connected component: {num_edges_largest}")


fraction_nodes = num_nodes_largest / G.number_of_nodes() *100
fraction_edges = num_edges_largest / G.number_of_edges() *100

print(f"Ratio of nodes from the biggest connected component: {fraction_nodes:.4f}%")
print(f"Ratio of edges from the biggest connected component: {fraction_edges:.4f}%")

