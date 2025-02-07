import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, efficientnet_b0, regnet_y_400mf
from tqdm import tqdm
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seeds for reproducibility
random.seed(28)
np.random.seed(28)
torch.manual_seed(28)
torch.cuda.manual_seed(28)
torch.backends.cudnn.deterministic = True


def get_dataloader(path: str, batch_size: int) -> DataLoader:
    """
    Create a DataLoader from an image dataset.

    :param path: Path to the dataset directory.
    :param batch_size: Batch size for DataLoader.
    :return: DataLoader instance.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True if "train" in path else False
    )


def calculate_metrics(labels_true, labels_pred):
    """
    Calculate accuracy, precision, recall, and F1 score.

    :param labels_true: True labels.
    :param labels_pred: Predicted labels.
    :return: Accuracy, Precision, Recall, F1 score.
    """
    acc = accuracy_score(labels_true, labels_pred)
    precision = precision_score(
        labels_true, labels_pred, average="weighted", zero_division=0
    )
    recall = recall_score(labels_true, labels_pred, average="weighted", zero_division=0)
    f1 = f1_score(labels_true, labels_pred, average="weighted", zero_division=0)
    return acc, precision, recall, f1


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """
    Train the model for one epoch.

    :param model: Model to be trained.
    :param dataloader: DataLoader for training.
    :param optimizer: Optimizer instance.
    :param loss_function: Loss function.
    :param device: Device (CPU or GPU).
    :return: Training loss.
    """
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate the model on the validation/test dataset.

    :param model: Model to be evaluated.
    :param dataloader: DataLoader for validation/test.
    :param loss_function: Loss function.
    :param device: Device (CPU or GPU).
    :return: Validation loss and metrics.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    loop = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    acc, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    return running_loss / len(dataloader), acc, precision, recall, f1


def train_and_evaluate(
    model: nn.Module,
    model_name: str,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    device: torch.device,
    epochs: int = 5,
):
    """
    Train and evaluate a model, logging results and saving the best model.

    :param model: Model to be trained and evaluated.
    :param model_name: Name of the model.
    :param train_dataloader: Training DataLoader.
    :param test_dataloader: Test DataLoader.
    :param device: Device (CPU or GPU).
    :param epochs: Number of epochs to train.
    """
    wandb.init(project="small-object-classifier", name=model_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    model.to(device)
    best_acc = float("-inf")

    # Initialize variables to accumulate the metrics
    (
        total_test_loss,
        total_test_acc,
        total_test_precision,
        total_test_recall,
        total_test_f1,
    ) = 0.0, 0.0, 0.0, 0.0, 0.0

    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, optimizer, loss_function, device)
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            model, test_dataloader, loss_function, device
        )

        # Accumulate test metrics
        total_test_loss += test_loss
        total_test_acc += test_acc
        total_test_precision += test_precision
        total_test_recall += test_recall
        total_test_f1 += test_f1

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            }
        )

        print(f"Epoch: {epoch+1:02}")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(
            f"\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc * 100:.2f}% | Val. Precision: {test_precision:.2f} | Val. Recall: {test_recall:.2f} | Val. F1: {test_f1:.2f}"
        )

        if best_acc < test_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")

    # Calculate averages for all metrics
    num_epochs = epochs
    avg_test_loss = total_test_loss / num_epochs
    avg_test_acc = total_test_acc / num_epochs
    avg_test_precision = total_test_precision / num_epochs
    avg_test_recall = total_test_recall / num_epochs
    avg_test_f1 = total_test_f1 / num_epochs

    # Save averages to CSV
    results = {
        "avg_test_loss": avg_test_loss,
        "avg_test_acc": avg_test_acc,
        "avg_test_precision": avg_test_precision,
        "avg_test_recall": avg_test_recall,
        "avg_test_f1": avg_test_f1,
    }

    df = pd.DataFrame([results])
    df.to_csv(f"{model_name}_average_results.csv", index=False)

    wandb.finish()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    train_dataloader = get_dataloader("images/train", batch_size)
    test_dataloader = get_dataloader("images/test", batch_size)

    train_and_evaluate(
        resnet18(weights="ResNet18_Weights.DEFAULT"),
        "ResNet18",
        train_dataloader,
        test_dataloader,
        device,
        10,
    )
    train_and_evaluate(
        efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT"),
        "EfficientNet_B0",
        train_dataloader,
        test_dataloader,
        device,
        10,
    )
    train_and_evaluate(
        regnet_y_400mf(weights="RegNet_Y_400MF_Weights.DEFAULT"),
        "RegNet_Y_400MF",
        train_dataloader,
        test_dataloader,
        device,
        10,
    )
