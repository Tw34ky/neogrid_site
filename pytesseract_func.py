import pytesseract
from PIL import Image
import os
import fitz
import pprint


def pdf_to_text(pdf_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
    text = ''
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pixmap = page.get_pixmap()
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            temp = pytesseract.image_to_string(image, config=r'--oem 1 --psm 3', lang='eng+rus')
            text += temp
    return text
