
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

from model import MNISTMLP


# ---------------------------------
# DOSYA VE MODEL AYARLARI
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "my_digits"
MODEL_PATH = BASE_DIR / "results" / "mnist_mlp.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize((0.1307,), (0.3081,))

# Fotoğraflardaki rakamların gerçek değerleri
EXPECTED_LABELS = {
    "digit_1.jpeg": 2,
    "digit_2.jpeg": 4,
    "digit_3.jpeg": 1,
    "digit_4.jpeg": 2,
    "digit_5.jpeg": 0,
    "digit_6.jpeg": 2,
    "digit_7.jpeg": 0,
    "digit_8.jpeg": 5,
    "digit_9.jpeg": 0,
    "digit_10.jpeg": 3,
    "digit_11.jpeg": 4,
}


# ---------------------------------
# GÖRÜNTÜYÜ HAZIRLA
# ---------------------------------

def prepare_image(image_path):
    # 1. Gri tonlamalı aç ve 500x500 sınırına çek
    image = Image.open(image_path).convert("L")
    image.thumbnail((500, 500), Image.Resampling.LANCZOS)
    
    # 2. Sadece ortaya odaklan (Kenarlardaki masayı at)
    w, h = image.size
    image = image.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
    
    # 3. Tersine çevir
    image = ImageOps.invert(image)
    
    # 4. Dinamik kontrast ve Eşikleme
    image = ImageOps.autocontrast(image, cutoff=2)
    image = image.point(lambda p: 255 if p > 128 else 0)
    
    # 5. Gürültü Temizleme
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    # 6. ÇİZGİLERİ KALINLAŞTIR (Resim büyükken yapıyoruz!)
    # Böylece 0'ların içi kapanmaz, sadece mürekkep gerçekçi bir şekilde dağılır.
    image = image.filter(ImageFilter.MaxFilter(5))
    
    # 7. Sınırları bul ve sadece rakamı kırp
    bbox = image.getbbox()
    if bbox is None:
        raise ValueError("Görüntüde rakam bulunamadı.")
    image = image.crop(bbox)
    
    # 8. Rakamı 20x20 alana orantılı sığdır
    ratio = 20.0 / max(image.width, image.height)
    new_w = max(1, int(image.width * ratio))
    new_h = max(1, int(image.height * ratio))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 9. Küçültme sonrası grileşen kenarları netleştir
    image = image.point(lambda p: 255 if p > 50 else 0)
    
    # 10. 28x28 siyah tuvalin ortasına yerleştir
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))
    
    # Önizlemeyi kaydet
    canvas.save(BASE_DIR / f"preview_{image_path.stem}.png")
    
    # 11. Tensor'a dönüştür
    tensor = transforms.ToTensor()(canvas)
    tensor = normalize(tensor)
    
    return tensor.unsqueeze(0)


# ---------------------------------
# MODELİ YÜKLE
# ---------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {MODEL_PATH}\n"
            "Önce main.py dosyasını çalıştır."
        )

    model = MNISTMLP().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    # Eski ve yeni kayıt biçimlerini destekle
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

        if "hidden_sizes" in checkpoint:
            model = MNISTMLP(
                hidden_sizes=tuple(checkpoint["hidden_sizes"])
            ).to(device)
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


# ---------------------------------
# TAHMİN YAP
# ---------------------------------

def predict_image(model, image_path):
    tensor = prepare_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted = probabilities.max(dim=1)

    return predicted.item(), confidence.item() * 100


# ---------------------------------
# TÜM GÖRÜNTÜLERİ TEST ET
# ---------------------------------

def main():
    if not IMAGE_DIR.exists():
        print(f"Fotoğraf klasörü bulunamadı: {IMAGE_DIR}")
        return

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    image_paths = [
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    ]

    # digit_1, digit_2, ..., digit_11 sıralaması
    image_paths.sort(
        key=lambda path: (
            int(path.stem.split("_")[-1])
            if path.stem.split("_")[-1].isdigit()
            else float("inf"),
            path.name.lower()
        )
    )

    if not image_paths:
        print("my_digits klasöründe görüntü bulunamadı.")
        return

    model = load_model()

    correct = 0
    evaluated = 0

    print("\nKendi el yazın için model tahminleri:\n")

    for image_path in image_paths:
        try:
            predicted, confidence = predict_image(model, image_path)

            print(f"Dosya: {image_path.name}")
            print(f"Tahmin edilen rakam: {predicted}")
            print(f"Modelin tahmin güveni: %{confidence:.2f}")

            expected = EXPECTED_LABELS.get(image_path.name)

            if expected is not None:
                evaluated += 1
                is_correct = predicted == expected

                if is_correct:
                    correct += 1

                print(f"Gerçek rakam: {expected}")
                print(f"Sonuç: {'Doğru' if is_correct else 'Yanlış'}")

            print("-" * 35)

        except Exception as error:
            print(f"{image_path.name} işlenemedi: {error}")
            print("-" * 35)

    if evaluated:
        print("\nEl yazısı test özeti")
        print(f"Doğru tahmin: {correct}/{evaluated}")
        print(f"Doğruluk: %{correct / evaluated * 100:.2f}")


if __name__ == "__main__":
    main()
