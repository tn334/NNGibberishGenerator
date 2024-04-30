import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = open("../Training Files/words_alpha.txt").readlines()
data = list(data)

# input_size = # size of input features (26)
# hidden_size = # number of features in the hidden state (512)
# num_layers = # number of LSTM layers (8)
# output_size = # size of the output (7)
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)