import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from engine import ModelTrainer

# Constants
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 1

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(f"Using device: {device}")

def main():
    # Create model
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=101)
    ).to(device)

    # Get the transforms
    transform = weights.transforms()
    # print(transform)

    # Dataset and DataLoader setup
    train_dataset = Food101(root='src', split='train', transform=transform, download=True)
    test_dataset = Food101(root='src', split='test', transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    params = ModelTrainer(model=model,
                    train_dataloader=train_loader,
                    test_dataloader=test_loader,
                    loss_fn=criterion,
                    optimizer=optimizer,
                    epochs=NUM_EPOCHS,
                    device=device)
    results= params.run()

if __name__ == '__main__':
    print(f"Using device: {device}")
    main()
