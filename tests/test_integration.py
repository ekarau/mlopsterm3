import pytest
import pandas as pd
import os
from src.features import preprocess_data  # Bu fonksiyonun senin projende var olduğunu varsayıyorum

# Not: Eğer src modülü bulunamazsa, testi çalıştırmadan önce
# export PYTHONPATH=$PYTHONPATH:. komutunu kullanman gerekebilir.

def test_data_ingestion_and_processing_integration(tmp_path):
    """
    Component Test: Veri okuma ve işleme mantığının entegrasyonunu test eder.
    Gerçek veritabanı yerine geçici bir CSV dosyası (mock data source) kullanılır.
    """
    # 1. Setup: Geçici bir veri seti oluştur (Mocking Data Source)
    d = {'feature1': ['A', 'B', 'A'], 'feature2': [10, 20, 10], 'target': [0, 1, 0]}
    df = pd.DataFrame(data=d)
    
    # Geçici klasöre kaydet
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    
    # 2. Execution: İşleme fonksiyonunu çağır (Senin src kodundaki mantık)
    # Burada preprocess_data fonksiyonunun bir dosya yolundan okuyup df döndürdüğünü varsayıyoruz
    # Kendi koduna göre bu kısmı güncellemelisin.
    try:
        processed_df = pd.read_csv(file_path) # Basit okuma testi
        # Veya: processed_df = preprocess_data(file_path)
    except Exception as e:
        pytest.fail(f"Veri kaynağı entegrasyonu başarısız oldu: {e}")

    # 3. Assertion: Veri bütünlüğü kontrolü
    assert not processed_df.empty, "İşlenen veri seti boş olmamalı"
    assert 'feature1' in processed_df.columns, "Beklenen sütunlar eksik"
    assert len(processed_df) == 3, "Veri satır sayısı korunmalı"