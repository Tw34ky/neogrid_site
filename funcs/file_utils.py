import os
import docx
import pymupdf
from striprtf.striprtf import rtf_to_text
from funcs.global_vars import SUPPORTED_FORMATS
from funcs.pytesseract_func import pdf_to_text  # используется для OCR
import data_base_lib


def parse_file(filepath: str):
    """Извлекает текст из файла поддерживаемого формата."""
    try:
        if filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

        elif filepath.endswith('.docx') or filepath.endswith('.doc'):
            doc = docx.Document(filepath)
            return ' '.join([para.text for para in doc.paragraphs])

        elif filepath.endswith('.pdf'):
            with pymupdf.open(filepath) as doc:
                content = ""
                for page in doc:
                    content += page.get_text()
                if len(content) == 0:
                    content = pdf_to_text(filepath)  # OCR
                return content

        elif filepath.endswith('.rtf'):
            with open(filepath, encoding='utf-8') as infile:
                content = infile.read()
                return rtf_to_text(content)
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def db_expansion(filename):
    """Обрабатывает один файл и добавляет его в базу данных."""
    retrieved_data = parse_file(filename)
    if retrieved_data:
        print(filename)
        retrieved_data = retrieved_data.replace('\n', ' ')
        retrieved_data = retrieved_data.replace('\xad', '')
        retrieved_data = retrieved_data.replace('\u00ad', '')
        retrieved_data = retrieved_data.replace('\N{SOFT HYPHEN}', '')
        params = []
        metadata = {'source': filename}
        data_base_lib.populate_database(params, retrieved_data, metadata)
        return retrieved_data
    return None


def format_file_size(size_in_bytes):
    """Форматирует размер файла в читаемый вид (Б, КБ, МБ и т.д.)."""
    import math
    if size_in_bytes == 0:
        return ''
    units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    unit_index = min(int(math.log(size_in_bytes, 1024)), len(units) - 1)
    size = size_in_bytes / (1024 ** unit_index)
    if size.is_integer() or size >= 10 or unit_index == 0:
        return f"{int(round(size))} {units[unit_index]}"
    else:
        return f"{size:.2f}".rstrip('0').rstrip('.') + f" {units[unit_index]}"