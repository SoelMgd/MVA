"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


# Loading

file_path = '/content/ALTEGRAD_LAB4/code/datasets/CA-HepTh.txt'
G = nx.read_edgelist(file_path)
components = list(nx.connected_components(G))

largest_component = max(components, key=len)
G_largest = G.subgraph(largest_component)



############## Task 3

def spectral_clustering(G, k):
    
    A = nx.to_numpy_array(G)

    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    D_inv = np.diag(1.0 / degrees)  
    Lrw = np.eye(len(G)) - D_inv @ A

    eigvals, eigvecs = eigs(Lrw, k=k, which='SM')  
    

    eigvecs = np.real(eigvecs)


    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(eigvecs)

    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}
    return clustering


############## Task 4

print('\nTask 4.\n')
clustering = spectral_clustering(G, 50)
print(clustering)


############## Task 5

def modularity(G, clustering):
    m = G.number_of_edges() 
    modularity_value = 0.0
    
    community_degrees = {community: 0 for community in set(clustering.values())}
    community_edges = {community: 0 for community in set(clustering.values())}
    
    for node in G.nodes():
        community_degrees[clustering[node]] += G.degree(node)
    
    for u, v in G.edges():
        if clustering[u] == clustering[v]:
            community_edges[clustering[u]] += 1
    
    for community in community_degrees:
        lc = community_edges[community]  
        dc = community_degrees[community] 
        modularity_value += (lc / m) - (dc / (2 * m)) ** 2
    
    return modularity_value



############## Task 6

print('\nTask 6.\n')



def random_clustering(G, k):
    clustering = {node: randint(0, k-1) for node in G.nodes()}
    return clustering



clustering_spectral = spectral_clustering(G_largest, 50)
clustering_random = random_clustering(G_largest, 50)

modularity_spectral = modularity(G_largest, clustering_spectral)
modularity_random = modularity(G_largest, clustering_random)

print(f"Spectral clustering modularity: {modularity_spectral:.4f}")
print(f"Random clustering modularity: {modularity_random:.4f}")







