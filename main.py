
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

from model import MNISTMLP
from train import train_one_epoch, evaluate


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Kullanılan cihaz:", device)

    # Veri dönüşümleri
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Eğitim ve test veri kümelerini yükle
    full_train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Eğitim verisini eğitim ve doğrulama olarak ayır
    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [50000, 10000],
        generator=generator
    )

    # DataLoader oluştur
    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Modeli oluştur
    model = MNISTMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    # Eğitim sonuçlarını sakla
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Modeli eğit
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

    training_time = time.time() - start_time

    # Test veri kümesi üzerinde son değerlendirme
    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, device
    )

    print("\n--- Eğitim Sonuçları ---")
    print(f"Eğitim süresi: {training_time:.2f} saniye")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Modeli kaydet
    torch.save(model.state_dict(), "results/mnist_mlp.pth")
    print("Model results/mnist_mlp.pth konumuna kaydedildi.")

    # 1. Eğitim ve doğrulama loss grafiği
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs_range,
        history["train_loss"],
        marker="o",
        label="Training Loss"
    )
    plt.plot(
        epochs_range,
        history["val_loss"],
        marker="o",
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/loss_graph.png")
    plt.close()

    # 2. Eğitim ve doğrulama accuracy grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs_range,
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy"
    )
    plt.plot(
        epochs_range,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_graph.png")
    plt.close()

    print("Grafikler results klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
