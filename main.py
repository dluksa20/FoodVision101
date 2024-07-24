import torch
import torchvision
from torch import nn
from torchvision import transforms

BATCH_SIZE = 32

# Set device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(f"Using device: {device}")




# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download the Food-101 dataset
train_dataset = torchvision.datasets.Food101(root='data', split='train', transform=transform, download=True)
test_dataset = torchvision.datasets.Food101(root='data', split='test', transform=transform, download=True)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Print dataset information
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")