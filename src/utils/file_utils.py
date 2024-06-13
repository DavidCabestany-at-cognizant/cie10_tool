import json
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os

def load_environment():
    load_dotenv(find_dotenv())
    with open('./config/gpt_config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

def save_to_json(extracted_fields, filename="extracted_data.json"):
    try:
        data = json.loads(extracted_fields)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return filename
    except Exception as e:
        return f"Error during JSON export: {e}"

def save_to_excel(extracted_fields, pdf_file_name, filename="extracted_data.xlsx"):
    try:
        data = json.loads(extracted_fields)
        df = pd.DataFrame([data])
        df.insert(0, 'NOMBRE_ARCHIVO', pdf_file_name)
        df.to_excel(filename, index=False)
        return filename
    except Exception as e:
        return f"Error during Excel export: {e}"
