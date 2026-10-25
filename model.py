
import torch
import torch.nn as nn


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = MNISTMLP()

    sample_images = torch.randn(64, 1, 28, 28)
    outputs = model(sample_images)

    print("Model başarıyla oluşturuldu.")
    print("Girdi boyutu:", sample_images.shape)
    print("Çıktı boyutu:", outputs.shape)
    print(model)
