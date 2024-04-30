import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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
        output = self.fc(lstm_out[:, -1])  # Use only the last output of the sequence
        return output

# Define your dataset class
class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word = self.data[idx].strip()
        num_characters = 26 # Assuming lowercase English alphabet
        tensor_data = torch.zeros(len(word), num_characters, dtype=torch.float32)
        for i, char in enumerate(word):
            tensor_data[i, ord(char) - ord('a')] = 1
        # Correctly generate the target as the next character in the sequence
        if idx < len(self.data) - 1:
            next_char = self.data[idx + 1][0] if len(self.data[idx + 1]) > 0 else 'a'
            target = torch.tensor(ord(next_char) - ord('a'), dtype=torch.long)
        else:
            # Assuming 'a' as the default next character for the last word
            target = torch.tensor(ord('a') - ord('a'), dtype=torch.long)
        return tensor_data, target
    
    @staticmethod
    def collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs = pad_sequence(inputs, batch_first=True)
        targets = torch.stack(targets)
        return inputs, targets

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
output_size = 26 # Define based on your output dimension
learning_rate = 0.001
batch_size = 32
num_epochs = 30
log_interval = 100

# Prepare your data and create DataLoader
data = open("../Output/5_letter_dict.txt", "r").readlines()
dataset = WordDataset(data)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

# Define your model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train your model
train(model, train_loader, criterion, optimizer, num_epochs)

# Once trained, use the model to generate new words
# Example: seed_input = torch.tensor([your_seed_input], dtype=torch.float32)
seed_input = torch.zeros(1, 1, input_size)
output = model(seed_input)

# Use the output to generate new words
def generate_word(model, seed_input, length):
    model.eval()  # Set the model to evaluation mode
    word = ''
    for _ in range(length):
        output = model(seed_input)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        next_char = chr(predicted.item() + ord('a'))
        word += next_char
        # Prepare the next input. The new input is the one-hot encoding of the predicted character
        seed_input = torch.zeros(1, 1, 26)
        seed_input[0, 0, predicted.item()] = 1
    return word

# Generate a word of length 5
print(generate_word(model, seed_input, 5))
