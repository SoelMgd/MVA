import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '/content/ALTEGRAD_LAB4/code/datasets/train_5500_coarse.label'
path_to_test_set = '/content/ALTEGRAD_LAB4/code/datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt

# Task 11
def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx, doc in enumerate(docs):
        G = nx.Graph()
        
        # Add nodes for each word
        for word in doc:
            G.add_node(vocab[word])
        
        # Add edges based on the window size
        for i in range(len(doc) - window_size + 1):
            window = doc[i:i + window_size]
            for j in range(len(window)):
                for k in range(j + 1, len(window)):
                    G.add_edge(vocab[window[j]], vocab[window[k]])
        
        graphs.append(G)
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Add placeholder labels if missing
def add_node_labels(graphs):
    for G in graphs:
        for node in G.nodes():
            if 'label' not in G.nodes[node]:
                G.nodes[node]['label'] = str(node)  # Add a default label (e.g., the node ID)
    return graphs

# Apply to both training and test graphs
G_train_nx = add_node_labels(G_train_nx)
G_test_nx = add_node_labels(G_test_nx)

G_train = list(graph_from_networkx(G_train_nx, node_labels_tag="label"))
G_test = list(graph_from_networkx(G_test_nx, node_labels_tag="label"))

# Initialize the Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram)

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)


#Task 13

# Train an SVM classifier and make predictions
clf = SVC(kernel="precomputed")
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)

# Evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using WL subtree kernel: {accuracy:.4f}")



#Task 14


from grakel.kernels import VertexHistogram

# Initialize and compute the Vertex Histogram kernel
gk_vh = VertexHistogram()
K_train_vh = gk_vh.fit_transform(G_train)
K_test_vh = gk_vh.transform(G_test)

# Train and evaluate with Vertex Histogram kernel
clf_vh = SVC(kernel="precomputed")
clf_vh.fit(K_train_vh, y_train)
y_pred_vh = clf_vh.predict(K_test_vh)

accuracy_vh = accuracy_score(y_test, y_pred_vh)
print(f"Accuracy using Vertex Histogram kernel: {accuracy_vh:.4f}")