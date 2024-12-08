"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""
import numpy as np

def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    ############## Task 1
    X_train = np.zeros((n_train, max_train_card), dtype=int)
    y_train = np.zeros(n_train, dtype=int)
    M = np.random.randint(1, max_train_card + 1, n_train)

    for i in range(n_train):
        M_i = M[i]
        digits = np.random.randint(1, 11, M_i)
        X_train[i, -M_i:] = digits
        y_train[i] = digits.sum()
    ##############

    return X_train, y_train

def create_test_dataset():
    
    ############## Task 2
    X_test = []
    y_test = []

    for card in range(5, 101, 5):
        n_samples = 10000
        samples = np.random.randint(1, 11, size=(n_samples, card))
        targets = np.sum(samples, axis=1)
        
        X_test.append(samples)
        y_test.append(targets)
    ##############

    return X_test, y_test