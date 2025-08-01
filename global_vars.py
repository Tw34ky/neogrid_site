import os

pytesseract_exe_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata"

SUPPORTED_FORMATS = ['rtf', 'pdf', 'docx', 'txt', 'doc']

SETTINGS = ['BASE_DIR', 'pytesseract_exe_path', 'tessdata_path', 'reset_data_boolean']

BASE_DIR = r"C:\Users\Тимофей\Documents" # os.path.abspath(os.path.expanduser('~'))

reset_data_boolean = False