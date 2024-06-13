import os
import json
import pandas as pd
from ..src.utils.file_utils import load_environment, save_to_json, save_to_excel

def test_load_environment(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test_key')
    config = load_environment()
    assert config['model'] == 'gpt-4o'
    assert os.getenv('OPENAI_API_KEY') == 'test_key'

def test_save_to_json(tmp_path):
    data = '{"key": "value"}'
    json_file = tmp_path / "test.json"
    save_to_json(data, filename=json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    assert saved_data["key"] == "value"

def test_save_to_excel(tmp_path):
    data = '{"key": "value"}'
    pdf_file_name = "test.pdf"
    excel_file = tmp_path / "test.xlsx"
    save_to_excel(data, pdf_file_name, filename=excel_file)
    df = pd.read_excel(excel_file)
    assert df.loc[0, 'key'] == "value"
    assert df.loc[0, 'NOMBRE_ARCHIVO'] == pdf_file_name
