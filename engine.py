import time
import torch
import torch.utils.data
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

class ModelTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 device: torch.device,
                 disable_bar: bool = False):
        """
        Initialize the ModelTrainer class.

        Args:
            model: PyTorch model to be trained and tested.
            train_dataloader: DataLoader instance for the training data.
            test_dataloader: DataLoader instance for the test data.
            optimizer: PyTorch optimizer to minimize loss.
            loss_fn: PyTorch loss function.
            epochs: Number of epochs for training.
            device: Device to run the training on (e.g., "cuda", "cpu").
            disable_bar: Flag to disable the progress bar.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.disable_bar = disable_bar
        self.results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "train_epoch_time": [],
            "test_epoch_time": []
        }

    def train_step(self, epoch: int) -> Tuple[float, float]:
        """
        Trains the model for a single epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple containing average training loss and accuracy for the epoch.
        """
        self.model.train()
        train_loss, train_acc = 0.0, 0.0

        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(self.train_dataloader),
                            desc=f"Training Epoch {epoch}",
                            total=len(self.train_dataloader),
                            disable=self.disable_bar)

        for batch, (X, y) in progress_bar:
            X, y = X.to(self.device), y.to(self.device)

            y_predicted = self.model(X)
            loss = self.loss_fn(y_predicted, y)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y_predicted_class = torch.argmax(torch.softmax(y_predicted, dim=1), dim=1)
            train_acc += (y_predicted_class == y).sum().item() / len(y_predicted_class)

            # Update progress bar
            progress_bar.set_postfix({
                "train_loss": train_loss / (batch + 1),
                "train_acc": train_acc / (batch + 1)
            })

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)
        return train_loss, train_acc

    def test_step(self, epoch: int) -> Tuple[float, float]:
        """
        Tests the model for a single epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple containing average test loss and accuracy for the epoch.
        """
        self.model.eval()
        test_loss, test_acc = 0.0, 0.0

        # Initialize tqdm progress bar
        progress_bar = tqdm(enumerate(self.test_dataloader),
                            desc=f"Testing Epoch {epoch}",
                            total=len(self.test_dataloader),
                            disable=self.disable_bar)

        with torch.no_grad():
            for batch, (X, y) in progress_bar:
                X, y = X.to(self.device), y.to(self.device)

                y_pred_log = self.model(X)
                loss = self.loss_fn(y_pred_log, y)
                test_loss += loss.item()

                y_pred_labels = y_pred_log.argmax(dim=1)
                test_acc += (y_pred_labels == y).sum().item() / len(y_pred_labels)

                # Update progress bar
                progress_bar.set_postfix({
                    "test_loss": test_loss / (batch + 1),
                    "test_acc": test_acc / (batch + 1)
                })

        test_loss /= len(self.test_dataloader)
        test_acc /= len(self.test_dataloader)
        return test_loss, test_acc

    def run(self) -> Dict[str, List[float]]:
        """
        Runs the training and testing process for the specified number of epochs.

        Returns:
            A dictionary containing lists of training and testing metrics and epoch times.
        """
        self.model.to(self.device)  # Ensure the model is on the correct device

        for epoch in tqdm(range(self.epochs), disable=self.disable_bar):
            # Training step
            train_epoch_start_time = time.time()
            train_loss, train_acc = self.train_step(epoch=epoch)
            train_epoch_end_time = time.time()
            train_epoch_time = train_epoch_end_time - train_epoch_start_time

            # Testing step
            test_epoch_start_time = time.time()
            test_loss, test_acc = self.test_step(epoch=epoch)
            test_epoch_end_time = time.time()
            test_epoch_time = test_epoch_end_time - test_epoch_start_time

            # Print metrics
            print(f"Epoch: {epoch + 1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_acc: {test_acc:.4f} | "
                  f"train_epoch_time: {train_epoch_time:.4f} | "
                  f"test_epoch_time: {test_epoch_time:.4f}")

            # Update results
            self.results["train_loss"].append(train_loss)
            self.results["train_acc"].append(train_acc)
            self.results["test_loss"].append(test_loss)
            self.results["test_acc"].append(test_acc)
            self.results["train_epoch_time"].append(train_epoch_time)
            self.results["test_epoch_time"].append(test_epoch_time)

        return self.results
