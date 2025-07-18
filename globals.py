import os

pytesseract_exe_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata"

SUPPORTED_FORMATS = ['rtf', 'pdf', 'docx', 'txt', 'doc']

SETTINGS = ['BASE_DIR', 'pytesseract_exe_path', 'tessdata_path', '']

BASE_DIR = os.path.abspath(os.path.expanduser('~'))

reset_data_boolean = False