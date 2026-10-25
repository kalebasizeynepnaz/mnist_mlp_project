
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

from model import MNISTMLP


# Proje klasörünün konumları
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "my_digits"
MODEL_PATH = BASE_DIR / "results" / "mnist_mlp.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daha önce eğittiğimiz modeli yükle
model = MNISTMLP().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

# MNIST ile aynı normalizasyon
normalize = transforms.Normalize((0.1307,), (0.3081,))


def prepare_image(image_path):
    # 1. Görüntüyü gri tonlamalı aç
    image = Image.open(image_path).convert("L")
    
    # 2. TEK ADIMDA TERSİNE ÇEVİRME VE EŞİKLEME (Thresholding)
    # Orijinal fotoğrafta kalem koyu (128'den küçük), kağıt ise açıktır.
    # Bu kod koyu renkli mürekkebi bembeyaz (255), açık renkli kağıdı simsiyah (0) yapar.
    image = image.point(lambda p: 255 if p < 130 else 0)
    
    # 3. Bounding Box (Kırpma)
    # Etrafta hiç kağıt lekesi kalmadığı için rakamı kusursuzca kenarlarından kesecek
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        
    # 4. ÇİZGİ KALINLAŞTIRMA (DILATION)
    # Rakamın ince kalem çizgilerini genişleterek MNIST'teki kalın fırça darbelerine benzetiyoruz.
    image = image.filter(ImageFilter.MaxFilter(3))
        
    # 5. Küçültme işlemi (Çizgilerin silinmemesi için LANCZOS yerine BICUBIC kullanıyoruz)
    image.thumbnail((20, 20), Image.Resampling.BICUBIC)
    
    # Küçültme sonrası kenarları yumuşayan pikselleri tekrar netleştiriyoruz
    image = image.point(lambda p: 255 if p > 50 else 0)

    # 6. 28x28 siyah tuvalin tam ortasına yerleştir
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    # 7. Tensor formatına çevir ve normalize et
    tensor = transforms.ToTensor()(canvas)
    tensor = normalize(tensor)

    return tensor.unsqueeze(0)


def main():
    if not IMAGE_DIR.exists():
        print("my_digits klasörü bulunamadı:", IMAGE_DIR)
        return

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        print("my_digits klasöründe desteklenen görüntü bulunamadı.")
        return

    print("Kendi el yazın için model tahminleri:\n")

    for image_path in image_paths:
        image_tensor = prepare_image(image_path).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_digit = outputs.argmax(dim=1).item()
            confidence = probabilities[0, predicted_digit].item() * 100

        print(f"Dosya: {image_path.name}")
        print(f"Tahmin edilen rakam: {predicted_digit}")
        print(f"Modelin tahmin güveni: %{confidence:.2f}")
        print("-" * 35)


if __name__ == "__main__":
    main()
