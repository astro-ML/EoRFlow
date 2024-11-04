import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(32 * 1 * 1, 128)  # Adjust output size if needed
        self.fc2 = nn.Linear(128, 7)  # Final summary size to pass to the flow

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (16, 8, 8)
        x = self.pool(F.relu(self.conv2(x)))  # (32, 1, 1)
        x = x.view(-1, 32 * 1 * 1)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Output summary for the flow
        return x
