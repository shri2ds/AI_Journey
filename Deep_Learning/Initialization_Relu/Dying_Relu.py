import torch
import torch.nn as nn

# 1. Generate Dummy Data (Circle pattern)
# We want the model to learn a simple non-linear shape
X = torch.rand(1000, 10)
Y = torch.randint(0, 2, (1000, 1)).float()

# 2. Define a Model with "Bad" Initialization
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(100, 1)

        # --- THE TRAP: Initialize weights to small negative numbers ---
        # This forces the first forward pass to be negative -> ReLU outputs 0.
        with torch.no_grad():
            self.layer1.weight.fill_(-0.1)
            self.layer1.bias.fill_(-0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)    # Inputs are all negative -> Output is ALL Zeros
        x = self.layer2(x)
        return x

model = BadNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
criterion = nn.BCEWithLogitsLoss()

# 3. Re-initialize properly
def init_weights_he(m):
    if isinstance(m, nn.Linear):
        # This matches the variance to the ReLU activation
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

print("\n--- Applying He Initialization ---")
model.apply(init_weights_he)

# 4. Training Loop (Watch the gradients)
print("Training with BadNet")
for epoch in range(15):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()

    # Check if gradients exist in Layer 1
    grad_magnitude = model.layer1.weight.grad.abs().sum().item()
    print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | Gradient Sum: {grad_magnitude}")

    optimizer.step()
