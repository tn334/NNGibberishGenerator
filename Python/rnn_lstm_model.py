import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define your RNN LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Use only the last output of the sequence
        return output

# Define your dataset class
class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert data point to tensors as needed
        # Example: return torch.tensor(self.data[idx], dtype=torch.float32)
        word = self.data[idx].strip()
        tensor_data = torch.zeros(len(word), dtype=torch.float32)
        for i, char in enumerate(word):
            tensor_data[i, ord(char) - ord('a')] = 1
        return tensor_data, tensor_data

# Define your training function
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

# Define hyperparameters
input_size = 26 # Define based on your features
hidden_size = 512 # Define based on your model complexity
output_size = 5 # Define based on your output dimension
learning_rate = 0.001
batch_size = 32
num_epochs = 10
log_interval = 100

# Prepare your data and create DataLoader
data = open("../Output/5_letter_dict.txt", "r").readlines()
dataset = WordDataset(data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train your model
train(model, train_loader, criterion, optimizer, num_epochs)

# Once trained, use the model to generate new words
# Example: seed_input = torch.tensor([your_seed_input], dtype=torch.float32)
seed_input = torch.zeros(1, input_size)
output = model(seed_input)
# Use the output to generate new words
