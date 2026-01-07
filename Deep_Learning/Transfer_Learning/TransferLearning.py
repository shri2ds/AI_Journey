import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import os, zipfile, requests


# 1. Setup Data
def download_data():
    if not os.path.exists("hymenoptera_data"):
        print("Downloading dataset...")
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        r = requests.get(url)
        with open("hymenoptera_data.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("hymenoptera_data.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Data downloaded and extracted.")

download_data()

# 2. Data Augmentation & Normalization
# Using ImageNet Mean/Std
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")

# 3. The Transfer Learning Model
print("\nLoading ResNet18...")
model = models.resnet18(weights='DEFAULT')

# STEP A: FREEZE the Body
# We iterate through parameters and turn off gradients except the layer4 and fc
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Sanity Check: Verify what is frozen
# Count parameters to ensure we aren't training the whole RESNET 
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nOur Custom Model Configuration:")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Layer 4 Frozen? {not list(model.layer4.parameters())[0].requires_grad}")
print(f"Layer 1 Frozen? {not list(model.layer1.parameters())[0].requires_grad}")

# STEP B: Replace the Head
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move to Device (GPU/MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# 4. Training (Only the Head Updates)
criterion = nn.CrossEntropyLoss()
# Optimize ONLY the final layer (model.fc.parameters)
optimiser = optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 1e-4}, # Low LR for body
    {'params': model.fc.parameters(), 'lr': 1e-3}      # Higher LR for head
], momentum=0.9)

print("Starting Training (Head Only)...")
for epoch in range(10): 
    model.eval()
    model.layer4.train()
    model.fc.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss = loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(image_datasets['train'])
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/10 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

print("Successfully leveraged existing RESNET & implemented Transfer learning")
