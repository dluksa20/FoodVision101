import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from engine import ModelTrainer
from model import Darknet19
from torch.utils.data import DataLoader

# Constants
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 1

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# print(f"Using device: {device}")


def main():
    # Create model instance and move to device
    model = Darknet19().to(device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((480, 744)),  # Resize images to match your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    # Dataset and DataLoader setup
    train_dataset = Food101(root='src', split='train', transform=transform, download=True)
    test_dataset = Food101(root='src', split='test', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    trainer = ModelTrainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device
    )

    trainer.run()

if __name__ == '__main__':
    print(f"Using device: {device}")
    main()