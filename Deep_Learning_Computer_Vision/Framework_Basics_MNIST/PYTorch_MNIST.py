import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: Convert raw images to Tensors and Normalize (0-1)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load Data (PyTorch handles the downloading)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Layer 1: 784 Inputs -> 128 Hidden neurons
        self.fc1 = nn.Linear(784, 128)
        # Layer 2: 128 Hidden -> 10 Output neurons
        self.fc2 = nn.Linear(128, 10)
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten image: (Batch, 28, 28) -> (Batch, 784)
        x = x.view(-1, 784)

        # Pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # We return Raw Logits 

model = SimpleNet()

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Starting Training...")
for epoch in range(3):  # Loop over dataset 3 times
    for batch_idx, (data, targets) in enumerate(train_loader):

        # 1. Forward Pass
        scores = model(data)
        loss = criterion(scores, targets)

        # 2. Backward Pass (The Magic Step)
        optimizer.zero_grad()  # Reset gradients from previous step
        loss.backward()  # Calculate ALL gradients automatically
        optimizer.step()  # Update weights: W = W - lr * grad

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
