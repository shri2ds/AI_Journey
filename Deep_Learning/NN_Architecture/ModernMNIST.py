import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Data Pipeline (Standardization is Key)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean/Std of MNIST dataset
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. The Modern Architecture
class ModernNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),

            # Layer 1
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),    # Stabilize Input
            nn.ReLU(),              # Activate
            nn.Dropout(0.2),        # Randomly kill 20% neurons

            # Layer 2
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Layer
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.feature_extractor(x)

model = ModernNet()

# 3. The Optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 4. Training Loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} | Loss: {loss.item():.6f}')

# 5. Test Loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct/len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {acc:.2f}%\n')

for epoch in range(1, 3):
    train(epoch)
    test()
