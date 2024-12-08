import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
#print(X_test.shape)
print(X_test[0].shape)
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        x_batch = X_test[i][j:j + batch_size]
        x_batch = torch.tensor(x_batch, dtype=torch.long).to(device)
        with torch.no_grad():
            output_deepsets = deepsets(x_batch).squeeze()
        y_pred_deepsets.append(output_deepsets)
        with torch.no_grad():
            output_lstm = lstm(x_batch).squeeze()
        y_pred_lstm.append(output_lstm)
        ##################
        
    y_pred_deepsets = torch.cat(y_pred_deepsets).detach().cpu().numpy().flatten()
    y_pred_lstm = torch.cat(y_pred_lstm).detach().cpu().numpy().flatten()
    y_true = y_test[i].flatten()
    
    acc_deepsets = accuracy_score(y_true, np.round(y_pred_deepsets))
    mae_deepsets = mean_absolute_error(y_true, y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    acc_lstm = accuracy_score(y_true, np.round(y_pred_lstm))
    mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
plt.figure(figsize=(10, 6))
plt.plot(cards, results['deepsets']['acc'], label='DeepSets Accuracy', marker='o')
plt.plot(cards, results['lstm']['acc'], label='LSTM Accuracy', marker='s')
plt.xlabel('Cardinality of Input Sets')
plt.ylabel('Accuracy')
plt.title('Model Accuracies vs Cardinality of Input Sets')
plt.legend()
plt.grid(True)
plt.savefig('accuracies.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(cards, results['deepsets']['mae'], label='DeepSets MAE', marker='o')
plt.plot(cards, results['lstm']['mae'], label='LSTM MAE', marker='s')
plt.xlabel('Cardinality of Input Sets')
plt.ylabel('Mean Absolute Error')
plt.title('Model MAE vs Cardinality of Input Sets')
plt.legend()
plt.grid(True)
plt.savefig('mae.png')  
plt.show()
##################