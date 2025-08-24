import os


pytesseract_exe_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_path = r'C:\Program Files\Tesseract-OCR\tessdata'
SETTINGS = ['BASE_DIR', 'pytesseract_exe_path', 'tessdata_path', 'reset_data_boolean', 'use_llm']
SUPPORTED_FORMATS = ['rtf', 'pdf', 'docx', 'txt', 'doc']

BASE_DIR = r'C:\Users\Timofey\Desktop\Sample Data' # os.path.abspath(os.path.expanduser('~'))

reset_data_boolean = False
use_llm = True
