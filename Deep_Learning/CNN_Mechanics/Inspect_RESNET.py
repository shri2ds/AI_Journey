import torch
import torch.nn as nn
import torchvision.models as models

# Concept: Max Pooling
# Let's prove it reduces dimensions and keeps the "strongest" feature.
def max_pooling():
    # Input Feature Map: (Batch=1, Channel=1, H=4, W=4)
    input_tensor = torch.tensor([
        [1.0, 3.0, 0.0, 2.0],
        [0.0, 8.0, 1.0, 0.0],
        [1.0, 0.0, 5.0, 2.0],
        [0.0, 1.0, 1.0, 0.0]
    ]).view(1, 1, 4, 4)

    # Pool with kernel 2x2, stride 2
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = pool(input_tensor)

    print(f"Input Shape: {input_tensor.shape}")
    print(f"Output Shape: {output.shape}")
    print("Output Values:\n", output.squeeze())

# Engineering: Inspecting ResNet
def inspect_resnet_internals():
    # Load Pre-trained weights (The 'Knowledge')
    # weights='DEFAULT' loads the best available ImageNet weights
    model = models.resnet18(weights='DEFAULT')

    # A. The State Dict (The Dictionary of Knowledge)
    print("Layer 1 Weights (First 2 filters):\n")
    # Key Name: 'conv1.weight' -> The very first scanning kernels
    print(model.state_dict()['conv1.weight'][0:2, 0, :, :])

    # B. The Architecture (Why we 'Chop off the Head')
    # ResNet ends with a Linear Layer: (fc): Linear(in_features=512, out_features=1000, bias=True)
    print("\nOriginal Final Layer:")
    print(model.fc)

    # C. Transfer Learning Prep (The Chop)
    # We replace the 1000-class output with OUR classes (e.g., 2 classes: Cat vs Dog)
    model.fc = nn.Linear(512, 2)
    print("\nModified Final Layer (Ready for Transfer Learning):")
    print(model.fc)

if __name__ == "__main__":
    max_pooling()
    inspect_resnet_internals()
