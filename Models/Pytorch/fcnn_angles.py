import torch
import torch.nn as nn


class EncoderAngles(nn.Module):
    def __init__(self):
        super(EncoderAngles, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(6, 8)  # Input layer with 6 neurons and hidden layer with 8 neurons
        self.fc3 = nn.Linear(8, 4)  # Output layer with 4 neurons

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation function if it's for regression)
        return x



