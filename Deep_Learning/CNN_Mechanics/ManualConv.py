import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load an Image (Grayscale)
def generate_synthetic_image():
    image = torch.zeros(1, 1, 10, 10) # Batch, Channel, H, W
    image[:, :, :, 5:] = 1.0 # Right half is white, Left is black (Vertical Edge!)
    return image

# Using synthetic image generated above 
input_image = generate_synthetic_image()

# 2. Define the Conv Layer
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)

# 3. Manually Set the Kernel (The "Sobel" Filter)
# This specific pattern detects VERTICAL edges.
sobel_kernel = torch.tensor([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
]).view(1, 1, 3, 3)     # (Out_Chan, In_Chan, H, W)

with torch.no_grad():
    conv.weight = nn.Parameter(sobel_kernel)

# 4. Apply Convolution
output = conv(input_image)

# 5. Visualize
print("Input Image Shape:", input_image.shape)
print("Output Feature Map Shape:", output.shape)

# Let's see the raw numbers.
# The middle column should light up because that's where 0 switches to 1.
print("\nOutput Feature Map Values:\n", output.squeeze())

# Plotting the input & output images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(input_image.squeeze(), cmap='gray')
ax[0].set_title("Input (Black & White)")
ax[1].imshow(output.detach().squeeze(), cmap='gray')
ax[1].set_title("Output (Vertical Edges Detected)")
plt.show()
