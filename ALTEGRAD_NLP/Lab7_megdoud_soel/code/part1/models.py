"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        ##################
        x = self.embedding(x)    # Shape: (batch_size, set_cardinality, embedding_dim)
        x = self.fc1(x)          # Shape: (batch_size, set_cardinality, hidden_dim)
        x = self.tanh(x)         # Shape: (batch_size, set_cardinality, hidden_dim)
        x = torch.sum(x, dim=1)  # Shape: (batch_size, hidden_dim)
        x = self.fc2(x)          # Shape: (batch_size, 1)
        ##################
        
        return x.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
    
        ##################
        x = self.embedding(x)       # Shape: (batch_size, set_cardinality, embedding_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: Shape (1, batch_size, hidden_dim)
        x = self.fc(h_n.squeeze(0)) # Shape: (batch_size, 1)
        ##################
        
        return x.squeeze()