import pytest
import pandas as pd


def test_data_ingestion_and_processing_integration(tmp_path):
    d = {'feature1': ['A', 'B', 'A'], 'feature2': [10, 20, 10], 'target': [0, 1, 0]}
    df = pd.DataFrame(data=d)

    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    try:
        processed_df = pd.read_csv(file_path)
    except Exception as e:
        pytest.fail(f"Veri kaynağı entegrasyonu başarısız oldu: {e}")

    assert not processed_df.empty, "İşlenen veri seti boş olmamalı"
    assert 'feature1' in processed_df.columns, "Beklenen sütunlar eksik"
    assert len(processed_df) == 3, "Veri satır sayısı korunmalı"
