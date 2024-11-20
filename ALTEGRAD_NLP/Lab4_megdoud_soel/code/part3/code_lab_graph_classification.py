"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

############## Task 7


#load Mutag dataset
def load_dataset():

  
    dataset = TUDataset(root='/content/ALTEGRAD_LAB4/code/datasets/MUTAG', name='MUTAG')
    Gs = [to_networkx(data) for data in dataset]

    y = [data.y.item() for data in dataset]
    return Gs, y


Gs,y = load_dataset()

#Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 8
# Compute the graphlet kernel

import random

def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    for i, G in enumerate(Gs_train):
        graphlet_counts = np.zeros(4)
        for _ in range(n_samples):
            
            nodes = np.random.choice(list(G.nodes()), size=3, replace=False)
            subgraph = G.subgraph(nodes)
            subgraph = nx.Graph(subgraph)
            
            # Check isomorphism against the graphlets
            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    graphlet_counts[j] += 1
        
        
        phi_train[i, :] = graphlet_counts

    phi_test = np.zeros((len(G_test), 4))
    
    for i, G in enumerate(Gs_test):
        graphlet_counts = np.zeros(4)
        for _ in range(n_samples):
            
            nodes = np.random.choice(list(G.nodes()), size=3, replace=False)
            subgraph = G.subgraph(nodes)
            subgraph = nx.Graph(subgraph)
            
            
            for j, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(subgraph, graphlet):
                    graphlet_counts[j] += 1
        
        
        phi_test[i, :] = graphlet_counts

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 9

K_train_gr, K_test_gr = graphlet_kernel(G_train, G_test)

############## Task 10


clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)
y_pred_sp = clf_sp.predict(K_test_sp)

accuracy_sp = accuracy_score(y_test, y_pred_sp)
print(f"Accuracy for Shortest Path Kernel: {accuracy_sp:.4f}")


clf_gr = SVC(kernel='precomputed')
clf_gr.fit(K_train_gr, y_train)
y_pred_gr = clf_gr.predict(K_test_gr)

accuracy_gr = accuracy_score(y_test, y_pred_gr)
print(f"Accuracy for Graphlet Kernel: {accuracy_gr:.4f}")
