#!/bin/bash

echo "🚀 MLOps Projesi Kurulumu Başlıyor..."

mkdir -p data/raw data/processed data/models data/interim

# 2. Docker Konteynerlerini Başlat (ve İnşa Et)
echo "📦 Docker imajları inşa ediliyor ve başlatılıyor..."
docker-compose up -d --build

# 3. Airflow'un ayağa kalkmasını bekle (Sağlık kontrolü)
echo "⏳ Airflow'un hazır olması bekleniyor (Bu işlem 30-60 sn sürebilir)..."
until docker exec final_version_airflow_1 airflow db check; do
  echo "   ... bekleniyor ..."
  sleep 5
done

# 4. Admin Kullanıcısını Otomatik Oluştur (Varsa hata vermez, geçer)
echo "👤 Admin kullanıcısı oluşturuluyor..."
docker exec final_version_airflow_1 airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

echo "✅ KURULUM TAMAMLANDI!"
echo "👉 Airflow: http://localhost:8080 (Giriş: admin / admin)"
echo "👉 API Docs: http://localhost:8000/docs"