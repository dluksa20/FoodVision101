import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader


# Path to the dataset
dataset_path = 'src/food-101'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check if dataset directory exists
if not os.path.exists(dataset_path):
    try:
        # Attempt to download the dataset
        train_dataset = Food101(root='src', split='train', transform=transform, download=True)
        test_dataset = Food101(root='src', split='test', transform=transform, download=True)
    except Exception as e:
        print(f"[INFO] An error occurred: {e}")
        print("[INFO] Falling back to manual dataset loading...")
else:
    print("[INFO] Dataset already exists. Skipping download.")
    train_dataset = Food101(root='src', split='train', transform=transform, download=False)
    test_dataset = Food101(root='src', split='test', transform=transform, download=False)
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
