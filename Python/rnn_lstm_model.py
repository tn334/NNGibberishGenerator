import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

with open("../Training Files/words_alpha.txt", "r") as input_file:
    lines = input_file.readlines()

# input_size = # size of input features (26)
# hidden_size = # number of features in the hidden state (512)
# num_layers = # number of LSTM layers (8)
# output_size = # size of the output (7)
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)

with open("../Output/5_letter_dict.txt", "w") as file_out:
    for line in lines:
        words = line.split()

        for word in words:
            if len(word) <= 5:
                file_out.write(word + "\n")