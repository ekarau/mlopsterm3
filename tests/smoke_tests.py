import requests
import time
import sys

# Konteynerın çalıştığı adres (docker-compose veya run ayarına göre değişebilir)
URL = "http://localhost:8000/predict"  # Portu kendi ayarına göre 5000 veya 8000 yap

# Test edilecek örnek veri (Homework temasına uygun High Cardinality örneği)
SAMPLE_PAYLOAD = {
    "features": {
        "category_feature": "some_high_cardinality_value",
        "numerical_feature": 123.45
    }
}


def wait_for_service(url, retries=5, delay=5):
    """Servisin ayağa kalkmasını bekler."""
    for i in range(retries):
        try:
            # Sağlık kontrolü veya basit bir istek
            response = requests.get(url.replace("/predict", "/"), timeout=2)  # Root endpoint kontrolü
            if response.status_code in [200, 404]:  # 404 de olsa server ayakta demektir
                print("Servis ayakta!")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Servise ulaşılamıyor, tekrar deneniyor... ({i + 1}/{retries})")
            time.sleep(delay)
    return False


def smoke_test():
    """API'ye tahmin isteği gönderir ve 200 OK arar."""
    if not wait_for_service(URL):
        print("HATA: Servis zaman aşımına uğradı, başlatılamadı.")
        sys.exit(1)

    try:
        response = requests.post(URL, json=SAMPLE_PAYLOAD)

        if response.status_code == 200:
            print("BAŞARILI: Smoke Test geçti! API 200 OK döndürdü.")
            print("Yanıt:", response.json())
            sys.exit(0)
        else:
            print(f"BAŞARISIZ: API hata kodu döndürdü: {response.status_code}")
            print("Detay:", response.text)
            sys.exit(1)

    except Exception as e:
        print(f"BAŞARISIZ: İstek sırasında hata oluştu: {e}")
        sys.exit(1)


if __name__ == "__main__":
    smoke_test()
