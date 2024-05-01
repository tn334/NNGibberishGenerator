import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

# Check if CUDA is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) # LSTM Layer
        self.fc = nn.Linear(hidden_size, output_size) # Sync Layer
    
    # forward pass
    def forward(self, x):
        lstm_out, _ = self.lstm(x) # perform forward pass through lstm layer
        output = self.fc(lstm_out[:, -1])  # Use only the last output of the sequence
        return output

class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        # length of dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        line = self.data[idx].strip().split(',') # split line into words and weights
        word = line[::2]  # Extract letters from the line
        weights = [int(w) for w in line[1::2]]  # Extract weights and convert to integers

        # initialize tensor for data
        tensor_data = torch.zeros(len(word), 26, dtype=torch.float32, device=device)

        # iterate over chars
        for i, char in enumerate(word):
            if char != '_': # check for stub char
                tensor_data[i, ord(char) - ord('a')] = weights[i]  # Use weight as frequency count
        if idx < len(self.data) - 1: # check for not last word
            next_word = self.data[idx + 1].strip().split(',')[::2] # get next word
            next_char = next_word[0] if next_word[0] != '_' else 'a' # get next char

            # set tensor target
            target = torch.tensor(ord(next_char) - ord('a'), dtype=torch.long, device=device)
        else: # if last word
            target = torch.tensor(ord('a') - ord('a'), dtype=torch.long, device=device) # set target as 'a'
        return tensor_data, target # return data and target
    
    # static method to collate batches
    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch) # unpack inputs and targets
        inputs = pad_sequence(inputs, batch_first=True) # pad sequences in batch
        targets = torch.stack(targets) # stack the targets
        return inputs, targets # return collated inputs and targets

# Define hyperparameters
input_size = 26       # dimensionality of input data
hidden_size = 600     # size of hidden state in LSTM
output_size = 26      # dimensionality of output data
learning_rate = 0.012 # learning rate for optimization
batch_size = 32       # batch size for training
num_epochs = 10       # num of training gens
log_interval = 95     # log interval for training

# read file data
data = open("../Output/5_letter_frequency_list_padded_comma_sep.txt", "r").readlines() 
dataset = WordDataset(data) # create dataset
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn) # create data loader

# start lstm model
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss() # set loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # set optimizer

# training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs): # iterate over epochs
        for batch_idx, (inputs, targets) in enumerate(train_loader): # iterate over batches
            optimizer.zero_grad() # zero gradients
            inputs, targets = inputs.to(device), targets.to(device) # move inputs, targets to gpu
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, targets) # compute loss
            loss.backward() # backward pass
            optimizer.step() # update parameters

            # display the loss
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")
        
        # Generate a word after each epoch
        generated_word = generate_word(model, seed_input, 5)
        print(f"\nGenerated word after epoch {epoch + 1}: {generated_word}\n")

# input used to generate a word
seed_input = torch.zeros(1, 1, input_size)

# Use the output to generate new words
def generate_word(model, seed_input, length):
    model.eval()  # Set the model to evaluation mode
    word = ''
    for _ in range(length): # generate word of specified length
        output = model(seed_input) # forward pass

        # calculate probabilities
        probabilities = F.softmax(output, dim=1).squeeze().detach().cpu().numpy()

        # choose next string index
        next_char_index = random.choices(range(len(probabilities)), probabilities)[0]
        
        # Convert the index to a character
        next_char = chr(next_char_index + ord('a'))
        
        word += next_char # append char to word
        seed_input = torch.zeros(1, 1, 26, device=device) # prepare next input 
        seed_input[0, 0, next_char_index] = 1 # set next input character
    return word

# display set hyperparameters used
print(f"""
    HYPERPARAMETERS 
      - hidden size: {hidden_size} 
      - learning rate: {learning_rate} 
      - batch size: {batch_size} 
      - total epochs: {num_epochs}
      - log interval: {log_interval}
      """)

# Train the model
train(model, train_loader, criterion, optimizer, num_epochs)
